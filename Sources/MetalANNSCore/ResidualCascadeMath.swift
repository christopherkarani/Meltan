import Accelerate
import Foundation

/// Deterministic PCA machinery for the exact cascade.
///
/// Split out of ResidualCascade.swift for file-size constraints. Members are
/// internal; `ResidualCascade` consumes only the final rotation + mean via
/// `pcaRotation`.
///
/// Numerical note: the covariance is formed in DOUBLE precision from a
/// materialized centered sample (`dsyrk`), then fully diagonalized with a
/// double-precision cyclic Jacobi sweep. An fp32-accumulated covariance
/// carries O(n·ε·‖v‖²) noise (~3 in this corpus's units) that both pollutes
/// the spectrum and tilts the basis, inflating projection-tail radii and
/// therefore survivor counts. The double-precision path costs ~2–3 s of
/// one-time build work (amortized, excluded from warm search latency).
enum ResidualCascadeMath {
    struct PCAResult {
        /// Column-major `dimensionCount × headWidth` orthonormal rotation.
        let rotation: [Float]
        /// DimensionCount-length corpus mean.
        let mean: [Float]
        /// Adaptive width chosen from the eigen-energy threshold.
        let headWidth: Int
        /// ‖μ‖².
        let meanNormSq: Float
    }

    /// Per-query values consumed by the bound computation.
    struct QueryBoundContext {
        let headDots: [Float]
        let tailNorm: Float
        let norm: Float
        let normSq: Float
        let dotMean: Float
    }

    /// Centers and projects a query using the cached PCA basis.
    static func prepareQueryContext(
        query: [Float], aux: ResidualCascade.BoundBuffer, dimensionCount: Int
    ) -> QueryBoundContext {
        var queryNormSq: Float = 0
        vDSP_dotpr(query, 1, query, 1, &queryNormSq, vDSP_Length(dimensionCount))

        var centeredQuery = [Float](repeating: 0, count: dimensionCount)
        for dimension in 0..<dimensionCount {
            centeredQuery[dimension] = query[dimension] - aux.meanVector[dimension]
        }
        let headWidth = aux.headWidth
        var queryHead = [Float](repeating: 0, count: headWidth)
        centeredQuery.withUnsafeBufferPointer { centeredBuffer in
            aux.rotation.withUnsafeBufferPointer { rotationBuffer in
                queryHead.withUnsafeMutableBufferPointer { headBuffer in
                    cblas_sgemv(
                        CblasColMajor, CblasTrans,
                        Int32(dimensionCount), Int32(headWidth), 1.0,
                        rotationBuffer.baseAddress!, Int32(dimensionCount),
                        centeredBuffer.baseAddress!, 1, 0.0,
                        headBuffer.baseAddress!, 1
                    )
                }
            }
        }
        var headEnergy: Float = 0
        vDSP_dotpr(queryHead, 1, queryHead, 1, &headEnergy, vDSP_Length(headWidth))
        var tailNormSq = queryNormSq - headEnergy
        if tailNormSq < 0 { tailNormSq = 0 }
        var dotMean: Float = 0
        vDSP_dotpr(query, 1, aux.meanVector, 1, &dotMean, vDSP_Length(dimensionCount))

        return QueryBoundContext(
            headDots: queryHead,
            tailNorm: sqrt(tailNormSq),
            norm: sqrt(queryNormSq),
            normSq: queryNormSq,
            dotMean: dotMean
        )
    }

    /// Mean + top-`recommendedWidth` eigenvectors of the sample covariance,
    /// computed in double precision via dsyrk + cyclic Jacobi. Deterministic.
    static func pcaRotation(
        corpus: UnsafePointer<Float>,
        rowCount: Int,
        dimensionCount: Int,
        sampleSize: Int,
        maxHeadWidth: Int
    ) -> PCAResult? {
        let strideStep = max(1, rowCount / sampleSize)
        let actualSample = min(sampleSize, (rowCount - 1) / strideStep + 1)
        let mean = sampleMean(
            corpus: corpus, strideStep: strideStep,
            sampleRows: actualSample, dimensionCount: dimensionCount
        )

        // ---- Materialize centered sample in Double (row-major rows×dim) ---
        var centeredSample = [Double](repeating: 0, count: actualSample * dimensionCount)
        var meanD = [Double](repeating: 0, count: dimensionCount)
        for dimension in 0..<dimensionCount {
            meanD[dimension] = Double(mean[dimension])
        }
        for sampleIndex in 0..<actualSample {
            let sourceBase = corpus + sampleIndex * strideStep * dimensionCount
            let targetBase = sampleIndex * dimensionCount
            for dimension in 0..<dimensionCount {
                centeredSample[targetBase + dimension] =
                    Double(sourceBase[dimension]) - meanD[dimension]
            }
        }

        // ---- Covariance C = Zᵀ·Z / n (double, symmetric dim×dim) ----------
        var covariance = [Double](repeating: 0, count: dimensionCount * dimensionCount)
        centeredSample.withUnsafeMutableBufferPointer { sampleBuf in
            covariance.withUnsafeMutableBufferPointer { covBuf in
                // RowMajor, Trans: C = Zᵀ·Z where Z is rows×dim row-major.
                cblas_dsyrk(
                    CblasRowMajor, CblasLower, CblasTrans,
                    Int32(dimensionCount), Int32(actualSample),
                    1.0, sampleBuf.baseAddress!, Int32(dimensionCount),
                    0.0, covBuf.baseAddress!, Int32(dimensionCount)
                )
            }
        }
        // Mirror lower triangle into upper and scale by 1/n.
        let normalizationScale = 1.0 / Double(max(actualSample, 1))
        covariance.withUnsafeMutableBufferPointer { covBuf in
            let base = covBuf.baseAddress!
            for row in 0..<dimensionCount {
                for col in 0...row {
                    let value = base[row * dimensionCount + col] * normalizationScale
                    base[row * dimensionCount + col] = value
                    base[col * dimensionCount + row] = value
                }
            }
        }

        // ---- Full double-precision Jacobi diagonalization -----------------
        var eigenValues = [Double](repeating: 0, count: dimensionCount)
        var eigenVectors = [Double](repeating: 0, count: dimensionCount * dimensionCount)
        jacobiEigenDecompositionDouble(
            matrix: &covariance, size: dimensionCount,
            outEigenValues: &eigenValues, outEigenVectors: &eigenVectors
        )

        // ---- Adaptive width from eigen-energy threshold --------------------
        let order = Array(0..<dimensionCount).sorted { eigenValues[$0] > eigenValues[$1] }
        let totalVariance = eigenValues.reduce(0, +)
        var cumulative: Double = 0
        var recommendedWidth = dimensionCount
        for rank in 0..<order.count where cumulative < (1.0 - 1e-6) * max(totalVariance, 1e-30) {
            cumulative += max(eigenValues[order[rank]], 0)
            recommendedWidth = rank + 1
        }
        recommendedWidth = max(2, min(maxHeadWidth, recommendedWidth))

        if ProcessInfo.processInfo.environment["METALANNS_RESIDUAL_STATS"] == "1" {
            let topValues = order.prefix(8).map { String(format: "%.4g", eigenValues[$0]) }
            let joined = topValues.joined(separator: ", ")
            FileHandle.standardError.write(Data(
                "[ResidualCascadeMath] gram eigenvalues (desc): \(joined); width=\(recommendedWidth)\n".utf8
            ))
        }
        // Guard: covariance collapsed (constant corpus) → not eligible.
        if eigenValues[order[0]] <= 0 { return nil }

        // ---- Assemble float rotation (column-major dim × width) ------------
        var rotation = [Float](repeating: 0, count: dimensionCount * recommendedWidth)
        for targetColumn in 0..<recommendedWidth {
            let sourceColumn = order[targetColumn]
            for row in 0..<dimensionCount {
                // Jacobi stores eigenvectors as row-major columns: V[row, col].
                rotation[targetColumn * dimensionCount + row] =
                    Float(eigenVectors[row * dimensionCount + sourceColumn])
            }
        }

        var meanNormSqDouble: Double = 0
        for value in mean {
            meanNormSqDouble += Double(value) * Double(value)
        }

        return PCAResult(
            rotation: rotation,
            mean: mean,
            headWidth: recommendedWidth,
            meanNormSq: Float(meanNormSqDouble)
        )
    }

    /// Mean of strided sample rows.
    static func sampleMean(
        corpus: UnsafePointer<Float>,
        strideStep: Int,
        sampleRows: Int,
        dimensionCount: Int
    ) -> [Float] {
        var mean = [Float](repeating: 0, count: dimensionCount)
        for sampleIndex in 0..<sampleRows {
            let rowBase = corpus + sampleIndex * strideStep * dimensionCount
            mean.withUnsafeMutableBufferPointer { meanBuf in
                vDSP_vadd(meanBuf.baseAddress!, 1, rowBase, 1,
                          meanBuf.baseAddress!, 1, vDSP_Length(dimensionCount))
            }
        }
        var meanScale = 1.0 / Float(sampleRows)
        mean.withUnsafeMutableBufferPointer { meanBuf in
            withUnsafePointer(to: &meanScale) { scalarPtr in
                vDSP_vsmul(meanBuf.baseAddress!, 1, scalarPtr,
                           meanBuf.baseAddress!, 1, vDSP_Length(dimensionCount))
            }
        }
        return mean
    }

    /// Cyclic Jacobi eigenvalue decomposition for a symmetric matrix
    /// (column-major). Produces unsorted eigenvalues and matching
    /// eigenvector columns. Deterministic.
    private static func jacobiEigenDecompositionDouble(
        matrix: inout [Double], size: Int,
        outEigenValues: inout [Double], outEigenVectors: inout [Double]
    ) {
        outEigenVectors = [Double](repeating: 0, count: size * size)
        for diagonal in 0..<size {
            outEigenVectors[diagonal * size + diagonal] = 1
        }
        let maxSweeps = 30
        for _ in 0..<maxSweeps {
            var offDiagonal: Double = 0
            for row in 0..<size {
                for col in (row + 1)..<size {
                    offDiagonal += matrix[row * size + col] * matrix[row * size + col]
                }
            }
            if offDiagonal < 1e-22 { break }
            for pivotRow in 0..<size {
                for pivotColumn in (pivotRow + 1)..<size {
                    let apq = matrix[pivotRow * size + pivotColumn]
                    if abs(apq) < 1e-14 { continue }
                    let app = matrix[pivotRow * size + pivotRow]
                    let aqq = matrix[pivotColumn * size + pivotColumn]
                    let theta = (aqq - app) / (2 * apq)
                    let sign: Double = theta >= 0 ? 1 : -1
                    let tangent = sign / (abs(theta) + (theta * theta + 1).squareRoot())
                    let cosine = 1 / (tangent * tangent + 1).squareRoot()
                    let sine = tangent * cosine
                    let rotation = JacobiRotation(
                        first: pivotRow, second: pivotColumn,
                        cosine: cosine, sine: sine
                    )
                    rotateSymmetricDouble(
                        matrix: &matrix, size: size, rotation: rotation
                    )
                    rotateEigenvectorsDouble(
                        vectors: &outEigenVectors, size: size, rotation: rotation
                    )
                }
            }
        }
        for diagonal in 0..<size {
            outEigenValues[diagonal] = matrix[diagonal * size + diagonal]
        }
    }

    private struct JacobiRotation {
        let first: Int
        let second: Int
        let cosine: Double
        let sine: Double
    }

    private static func rotateSymmetricDouble(
        matrix: inout [Double], size: Int, rotation: JacobiRotation
    ) {
        let first = rotation.first
        let second = rotation.second
        let cosine = rotation.cosine
        let sine = rotation.sine
        for index in 0..<size {
            let aFirst = matrix[index * size + first]
            let aSecond = matrix[index * size + second]
            matrix[index * size + first] = cosine * aFirst - sine * aSecond
            matrix[index * size + second] = sine * aFirst + cosine * aSecond
        }
        for index in 0..<size {
            let aFirst = matrix[first * size + index]
            let aSecond = matrix[second * size + index]
            matrix[first * size + index] = cosine * aFirst - sine * aSecond
            matrix[second * size + index] = sine * aFirst + cosine * aSecond
        }
    }

    private static func rotateEigenvectorsDouble(
        vectors: inout [Double], size: Int, rotation: JacobiRotation
    ) {
        let first = rotation.first
        let second = rotation.second
        let cosine = rotation.cosine
        let sine = rotation.sine
        for index in 0..<size {
            let vFirst = vectors[index * size + first]
            let vSecond = vectors[index * size + second]
            vectors[index * size + first] = cosine * vFirst - sine * vSecond
            vectors[index * size + second] = sine * vFirst + cosine * vSecond
        }
    }
}
