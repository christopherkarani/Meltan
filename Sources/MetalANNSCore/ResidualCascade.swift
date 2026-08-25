import Accelerate
import Foundation
import Metal
import simd
/// Exact nearest-neighbor cascade for large corpora built on PCA projections
/// with lossless verification bounds.
///
/// Motivation: residual-norm and quantized-code bounds cannot separate
/// neighbors along the same low-dimensional manifold, while an exact fp32
/// projection onto the top eigen directions concentrates dot-product energy
/// so a short projection plus a Cauchy–Schwarz tail term yields near-exact
/// bounds. The head width adapts to the measured eigen spectrum (smallest
/// prefix capturing nearly all sampled variance), clamped to [2, 48].
///
/// Per row (cached per backing MTL buffer):
///   - `proj[i]`     exact fp32 projection `Rᵀ(v − μ)` (adaptive width w)
///   - `vDotMu`      precomputed `v·μ`
///   - `rowNormSq`   exact `‖v‖²`
///   - `tailNorm`    `‖(v − μ) − R·Rᵀ(v − μ)‖` (Cauchy–Schwarz tail radius,
///                   derived via Pythagoras: `√(‖z‖² − ‖proj‖²)`)
///
/// Dot-product identity with centered vectors `x = q − μ`, `y = v − μ`:
///   q·v = x·y + q·μ + v·μ − ‖μ‖²,   x·y = proj_x·proj_y + x_t·y_t
///   x_t·y_t ≤ ‖x_t‖·tailNorm            (Cauchy–Schwarz)
///
/// All terms are inflated upward (sign-safely) by generous slack so fp
/// rounding can never invalidate the similarity upper bound.
///
/// Search protocol (exact on every path):
///   1. Per-row similarity upper bounds → distance lower bounds lb(row).
///   2. Rescore exactly m' rows with the smallest bounds ("seeds").
///   3. cutoff = k-th smallest true seed distance. Any row NOT rescored has
///      lb(row) ≥ cutoff ⇒ true distance ≥ cutoff, while the seeds supply
///      k rows at distance ≤ cutoff. Pruned rows can never beat the k-th
///      result (up to exact ties, which may reorder).
///   4. Rescore every remaining row with lb(row) < cutoff ("survivors").
///   5. Return the top-k over all rescored rows.
///
/// Loose bounds merely grow the survivor set (degrading toward brute force);
/// they can never change returned neighbors beyond tie order.
// pi-lens-ignore: type_body_length
enum ResidualCascade {
    /// Below this size the existing tiers already win; keep them.
    static let minVectorCount = 200_000

    /// Upper bound on adaptive projection width.
    static let maxHeadWidth = 48

    /// Generous relative slack applied to every derived bound.
    static let boundSlack: Float = 1e-4
    static let boundAbsSlack: Float = 1e-6

    // MARK: - Aux buffer

    final class BoundBuffer: @unchecked Sendable {
        let rowCount: Int
        let dimensionCount: Int
        let headWidth: Int
        let meanNormSq: Float
        let avgTailNorm: Float
        let avgRowNorm: Float

        /// Planar sections of one allocation (64-byte aligned offsets):
        /// proj[rowCount×headWidth] | vDotMu | rowNormSq | tailNorm
        private let storage: UnsafeMutableRawPointer
        let projBase: UnsafePointer<Float>
        let vDotMuPlane: UnsafePointer<Float>
        let rowNormSqPlane: UnsafePointer<Float>
        let tailNormPlane: UnsafePointer<Float>
        /// Column-major `dimensionCount × headWidth` rotation.
        let rotation: [Float]
        /// Corpus mean vector (`dimensionCount`).
        let meanVector: [Float]

        final class GPUPlanes {
            let deviceID: ObjectIdentifier
            let projection: MTLBuffer
            let vDotMu: MTLBuffer
            let rowNormSq: MTLBuffer
            let tailNorm: MTLBuffer

            init(
                deviceID: ObjectIdentifier,
                projection: MTLBuffer,
                vDotMu: MTLBuffer,
                rowNormSq: MTLBuffer,
                tailNorm: MTLBuffer
            ) {
                self.deviceID = deviceID
                self.projection = projection
                self.vDotMu = vDotMu
                self.rowNormSq = rowNormSq
                self.tailNorm = tailNorm
            }
        }

        private let gpuLock = NSLock()
        private var gpuPlanesCache: GPUPlanes?

        init(
            rowCount: Int,
            dimensionCount: Int,
            headWidth: Int,
            meanNormSq: Float,
            avgTailNorm: Float,
            avgRowNorm: Float,
            storage: UnsafeMutableRawPointer,
            projBase: UnsafePointer<Float>,
            vDotMuPlane: UnsafePointer<Float>,
            rowNormSqPlane: UnsafePointer<Float>,
            tailNormPlane: UnsafePointer<Float>,
            rotation: [Float],
            meanVector: [Float]
        ) {
            self.rowCount = rowCount
            self.dimensionCount = dimensionCount
            self.headWidth = headWidth
            self.meanNormSq = meanNormSq
            self.avgTailNorm = avgTailNorm
            self.avgRowNorm = avgRowNorm
            self.storage = storage
            self.projBase = projBase
            self.vDotMuPlane = vDotMuPlane
            self.rowNormSqPlane = rowNormSqPlane
            self.tailNormPlane = tailNormPlane
            self.rotation = rotation
            self.meanVector = meanVector
        }

        deinit {
            storage.deallocate()
        }

        /// Publishes shared Metal copies of the cached planes once. The
        /// CPU pointers remain the source of truth for synchronous fallback;
        /// the GPU copies are used only by the async bound pass.
        func gpuPlanes(device: MTLDevice) -> GPUPlanes? {
            let deviceID = ObjectIdentifier(device)
            gpuLock.lock()
            defer { gpuLock.unlock() }
            if let cached = gpuPlanesCache, cached.deviceID == deviceID {
                return cached
            }

            let floatBytes = MemoryLayout<Float>.stride
            guard let projection = device.makeBuffer(
                length: rowCount * headWidth * floatBytes, options: .storageModeShared
            ), let vDotMu = device.makeBuffer(
                length: rowCount * floatBytes, options: .storageModeShared
            ), let rowNormSq = device.makeBuffer(
                length: rowCount * floatBytes, options: .storageModeShared
            ), let tailNorm = device.makeBuffer(
                length: rowCount * floatBytes, options: .storageModeShared
            ) else { return nil }

            // Store projections column-major for the GPU: neighboring threads
            // then read neighboring rows for each PCA component, producing
            // coalesced loads (the CPU cache remains row-major).
            let gpuProjection = projection.contents().assumingMemoryBound(to: Float.self)
            DispatchQueue.concurrentPerform(iterations: headWidth) { column in
                let destination = gpuProjection + column * rowCount
                for row in 0..<rowCount {
                    destination[row] = projBase[row * headWidth + column]
                }
            }
            vDotMu.contents().copyMemory(
                from: UnsafeRawPointer(vDotMuPlane), byteCount: rowCount * floatBytes
            )
            rowNormSq.contents().copyMemory(
                from: UnsafeRawPointer(rowNormSqPlane), byteCount: rowCount * floatBytes
            )
            tailNorm.contents().copyMemory(
                from: UnsafeRawPointer(tailNormPlane), byteCount: rowCount * floatBytes
            )

            let planes = GPUPlanes(
                deviceID: deviceID, projection: projection,
                vDotMu: vDotMu, rowNormSq: rowNormSq, tailNorm: tailNorm
            )
            gpuPlanesCache = planes
            return planes
        }
    }

    // MARK: - Cache (mirrors Int8CodeCache semantics)

    struct Key: Hashable {
        let bufferID: ObjectIdentifier
        let bufferLength: Int
        let rowCount: Int
        let dimensionCount: Int
    }

    final class Store: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [Key: BoundBuffer] = [:]
        private var order: [Key] = []
        private let capacity = 8

        func get(_ key: Key) -> BoundBuffer? {
            lock.lock()
            defer { lock.unlock() }
            return entries[key]
        }

        func store(_ value: BoundBuffer, for key: Key) {
            lock.lock()
            defer { lock.unlock() }
            if entries[key] == nil {
                order.append(key)
                while order.count > capacity {
                    let evicted = order.removeFirst()
                    entries[evicted] = nil
                }
            }
            entries[key] = value
        }

        func invalidate(bufferID: ObjectIdentifier) {
            lock.lock()
            defer { lock.unlock() }
            let doomed = order.filter { $0.bufferID == bufferID }
            for key in doomed {
                entries[key] = nil
            }
            order.removeAll { $0.bufferID == bufferID }
        }
    }

    static let store = Store()

    static func invalidate(buffer: MTLBuffer) {
        store.invalidate(bufferID: ObjectIdentifier(buffer))
    }

    // MARK: - Build

    /// Builds the PCA-projection aux structure: adaptive-width rotation from
    /// subspace iteration, then per-row exact projections, scalar planes,
    /// and tail radii via Pythagoras (`tail² = ‖z‖² − ‖proj‖²`).
    static func build(
        corpus: UnsafePointer<Float>,
        rowCount: Int,
        dimensionCount: Int,
        key: Key
    ) -> BoundBuffer? {
        guard rowCount >= minVectorCount, dimensionCount > 0 else { return nil }

        let sampleSize = min(rowCount, 65_536)
        guard let pca = ResidualCascadeMath.pcaRotation(
            corpus: corpus, rowCount: rowCount, dimensionCount: dimensionCount,
            sampleSize: sampleSize, maxHeadWidth: maxHeadWidth
        ) else { return nil }
        let width = pca.headWidth

        // Allocation: proj block + three scalar planes.
        let alignment = 64
        func padded(_ bytes: Int) -> Int { (bytes + alignment - 1) / alignment * alignment }
        let projBytes = padded(rowCount * width * MemoryLayout<Float>.stride)
        let planeBytes = padded(rowCount * MemoryLayout<Float>.stride)
        let totalBytes = projBytes + 3 * planeBytes
        guard let raw = malloc(totalBytes) else { return nil }
        let projPtr = raw.assumingMemoryBound(to: Float.self)
        let vDotMuPtr = (raw + projBytes).assumingMemoryBound(to: Float.self)
        let rowNormSqPtr = (raw + projBytes + planeBytes).assumingMemoryBound(to: Float.self)
        let tailNormPtr = (raw + projBytes + 2 * planeBytes).assumingMemoryBound(to: Float.self)

        // Augmented rotation [R | μ̂] in ROW-major (dim × (width+1)) so the
        // chunked fill is one RowMajor sgemm per chunk. μ̂ = μ (unnormalized):
        // projecting z = v − μ against it yields z·μ directly.
        var rotationAugmented = [Float](repeating: 0, count: dimensionCount * (width + 1))
        for dimension in 0..<dimensionCount {
            for column in 0..<width {
                rotationAugmented[dimension * (width + 1) + column] =
                    pca.rotation[column * dimensionCount + dimension]
            }
            rotationAugmented[dimension * (width + 1) + width] = pca.mean[dimension]
        }
        // Constant correction row: μᵀ·R_aug (subtracted from every product).
        // RowMajor Trans: y = Aᵀ·μ has length (width+1) — must be Trans,
        // NoTrans would write dimensionCount floats into a (width+1) array.
        var meanProjection = [Float](repeating: 0, count: width + 1)
        rotationAugmented.withUnsafeBufferPointer { rotBuf in
            pca.mean.withUnsafeBufferPointer { meanBuf in
                cblas_sgemv(
                    CblasRowMajor, CblasTrans,
                    Int32(dimensionCount), Int32(width + 1), 1.0,
                    rotBuf.baseAddress!, Int32(width + 1),
                    meanBuf.baseAddress!, 1, 0.0, &meanProjection, 1
                )
            }
        }

        let workers = ProcessInfo.processInfo.activeProcessorCount
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let start = (rowCount * worker) / workers
            let end = (rowCount * (worker + 1)) / workers
            guard start < end else { return }

            var chunkStart = start
            while chunkStart < end {
                let rows = min(2_048, end - chunkStart)
                fillChunk(
                    corpus: corpus + chunkStart * dimensionCount,
                    rows: rows,
                    dimensionCount: dimensionCount,
                    plan: ChunkFillPlan(
                        rotationAugmented: rotationAugmented,
                        meanProjection: meanProjection,
                        meanNormSq: pca.meanNormSq,
                        width: width
                    ),
                    outputs: ChunkFillOutputs(
                        proj: projPtr + chunkStart * width,
                        vDotMu: vDotMuPtr + chunkStart,
                        rowNormSq: rowNormSqPtr + chunkStart,
                        tailNorm: tailNormPtr + chunkStart
                    )
                )
                chunkStart += rows
            }
        }

        // Diagnostics: average tail norm vs average row norm.
        var avgTailNorm: Float = 0
        var avgRowNorm: Float = 0
        for rowIndex in stride(from: 0, to: rowCount, by: max(1, rowCount / 10_000)) {
            avgTailNorm += tailNormPtr[rowIndex]
            avgRowNorm += sqrt(max(rowNormSqPtr[rowIndex], 0))
        }
        let sampleCount = Float(max(1, rowCount / max(1, rowCount / 10_000)))
        avgTailNorm /= sampleCount
        avgRowNorm /= sampleCount

        let buffer = BoundBuffer(
            rowCount: rowCount,
            dimensionCount: dimensionCount,
            headWidth: width,
            meanNormSq: pca.meanNormSq,
            avgTailNorm: avgTailNorm,
            avgRowNorm: avgRowNorm,
            storage: raw,
            projBase: projPtr,
            vDotMuPlane: vDotMuPtr,
            rowNormSqPlane: rowNormSqPtr,
            tailNormPlane: tailNormPtr,
            rotation: pca.rotation,
            meanVector: pca.mean
        )
        store.store(buffer, for: key)
        return buffer
    }

    /// Immutable per-build inputs for chunk fills (bundled to keep the
    /// function signature small).
    struct ChunkFillPlan {
        let rotationAugmented: [Float]
        let meanProjection: [Float]
        let meanNormSq: Float
        let width: Int
    }

    struct ChunkFillOutputs {
        let proj: UnsafeMutablePointer<Float>
        let vDotMu: UnsafeMutablePointer<Float>
        let rowNormSq: UnsafeMutablePointer<Float>
        let tailNorm: UnsafeMutablePointer<Float>
    }

    /// Fills projections and scalar planes for a contiguous chunk.
    /// One RowMajor sgemm computes `[proj | z·μ]`; tails come from Pythagoras.
    private static func fillChunk(
        corpus: UnsafePointer<Float>,
        rows: Int,
        dimensionCount: Int,
        plan: ChunkFillPlan,
        outputs: ChunkFillOutputs
    ) {
        let augmentedColumns = plan.width + 1
        var products = [Float](repeating: 0, count: rows * augmentedColumns)

        products.withUnsafeMutableBufferPointer { productsBuf in
            plan.rotationAugmented.withUnsafeBufferPointer { rotBuf in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(rows), Int32(augmentedColumns), Int32(dimensionCount),
                    1.0,
                    corpus, Int32(dimensionCount),
                    rotBuf.baseAddress!, Int32(augmentedColumns),
                    0.0, productsBuf.baseAddress!, Int32(augmentedColumns)
                )
            }
        }

        let width = plan.width
        let meanProjection = plan.meanProjection
        let meanNormSq = plan.meanNormSq
        let outProj = outputs.proj
        let outVDotMu = outputs.vDotMu
        let outRowNormSq = outputs.rowNormSq
        let outTailNorm = outputs.tailNorm

        for rowIndex in 0..<rows {
            let productBase = rowIndex * augmentedColumns
            // Subtract the constant μᵀ·R_aug row: products become z-projections.
            var headEnergy: Float = 0
            for column in 0..<width {
                let value = products[productBase + column] - meanProjection[column]
                outProj[rowIndex * width + column] = value
                headEnergy += value * value
            }
            let zDotMu = products[productBase + width] - meanProjection[width]

            var rowNormSq: Float = 0
            let rowBase = corpus + rowIndex * dimensionCount
            vDSP_dotpr(rowBase, 1, rowBase, 1, &rowNormSq, vDSP_Length(dimensionCount))
            outRowNormSq[rowIndex] = rowNormSq
            outVDotMu[rowIndex] = zDotMu + meanNormSq

            // ‖z‖² = ‖v‖² − 2·v·μ + ‖μ‖²; tail² = ‖z‖² − ‖proj‖².
            let centeredNormSq = rowNormSq - 2 * (zDotMu + meanNormSq) + meanNormSq
            var tailSq = centeredNormSq - headEnergy
            if tailSq < 0 { tailSq = 0 }
            outTailNorm[rowIndex] = sqrt(tailSq)
        }
    }

    // MARK: - Search

    /// Returns nil when ineligible; callers fall back to the flat GPU tier.
    static func search(
        query: [Float],
        vectors: any VectorStorage,
        neighborTotal: Int,
        metric: Metric
    ) -> [SearchResult]? {
        guard let vectorBuffer = vectors as? VectorBuffer,
            !vectors.isFloat16,
            let corpus = vectorBuffer.floatPointer.baseAddress
        else { return nil }
        let rowCount = vectors.count
        let dimensionCount = vectors.dim
        guard rowCount >= minVectorCount, dimensionCount > 0,
            query.count == dimensionCount, neighborTotal > 0, metric != .hamming
        else { return nil }

        let cacheKey = Key(
            bufferID: ObjectIdentifier(vectorBuffer.buffer),
            bufferLength: vectorBuffer.buffer.length,
            rowCount: rowCount,
            dimensionCount: dimensionCount
        )
        let aux: BoundBuffer
        if let cached = store.get(cacheKey), cached.rowCount >= rowCount {
            aux = cached
        } else {
            guard let built = build(
                corpus: corpus, rowCount: rowCount,
                dimensionCount: dimensionCount, key: cacheKey
            ) else { return nil }
            aux = built
        }

        let effectiveK = min(neighborTotal, FlatGPUSearch.maxTopK, rowCount)
        guard effectiveK > 0 else { return nil }

        // ---- Query preparation --------------------------------------------
        var queryNormSq: Float = 0
        vDSP_dotpr(query, 1, query, 1, &queryNormSq, vDSP_Length(dimensionCount))

        // Degenerate zero-norm query under cosine: every distance finalizes
        // to exactly 1.0, so ties resolve to the lowest ids — identical
        // outcome without scanning.
        if metric == .cosine && queryNormSq < 1e-20 {
            return lowestIdsResult(count: rowCount, take: effectiveK)
        }

        let boundContext = ResidualCascadeMath.prepareQueryContext(query: query, aux: aux, dimensionCount: dimensionCount)

        let statsEnabled = ProcessInfo.processInfo.environment["METALANNS_RESIDUAL_STATS"] == "1"
        var phaseTimings: [(String, Double)] = []
        func recordPhase(_ name: String, _ start: DispatchTime) {
            if statsEnabled {
                phaseTimings.append((name, Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000))
            }
        }

        let searchStart = DispatchTime.now()
        // ---- Phase 1: distance lower bounds --------------------------------
        let phase1Start = DispatchTime.now()
        var lowerBounds = [Float](repeating: 0, count: rowCount)
        lowerBounds.withUnsafeMutableBufferPointer { boundsBuf in
            computeBounds(
                context: boundContext,
                aux: aux,
                metric: metric,
                outLowerBounds: boundsBuf.baseAddress!
            )
        }
        recordPhase("p1_bounds", phase1Start)

        let results = resolveTopK(ResolveInput(
            lowerBounds: lowerBounds,
            corpus: corpus,
            dimensionCount: dimensionCount,
            query: query,
            queryNormSq: queryNormSq,
            aux: aux,
            metric: metric,
            effectiveK: effectiveK,
            rowCount: rowCount,
            recordPhase: recordPhase,
            statsEnabled: statsEnabled
        ))
        if statsEnabled {
            let summary = phaseTimings.map {
                String(format: "%@=%.2fms", $0.0, $0.1)
            }.joined(separator: " ")
            let preludeMs = Double(
                phase1Start.uptimeNanoseconds - searchStart.uptimeNanoseconds
            ) / 1_000_000
            let statsLine = String(
                format: "[ResidualCascade] phases: prelude=%.2fms %@\n",
                preludeMs, summary
            )
            FileHandle.standardError.write(Data(statsLine.utf8))
        }
        return results
    }

    // MARK: - Bound computation

    private static func computeBounds(
        context: ResidualCascadeMath.QueryBoundContext,
        aux: BoundBuffer,
        metric: Metric,
        outLowerBounds: UnsafeMutablePointer<Float>
    ) {
        let totalRows = aux.rowCount
        let headWidth = aux.headWidth
        let slack = boundSlack
        let absoluteSlack = boundAbsSlack
        let meanNormSq = aux.meanNormSq
        let workers = ProcessInfo.processInfo.activeProcessorCount

        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let start = (totalRows * worker) / workers
            let end = (totalRows * (worker + 1)) / workers
            guard start < end else { return }

            let proj = aux.projBase + start * headWidth
            let vDotMuPlane = aux.vDotMuPlane + start
            let rowNormSqPlane = aux.rowNormSqPlane + start
            let tailNormPlane = aux.tailNormPlane + start
            let out = outLowerBounds + start
            let queryHead = context.headDots

            for offset in 0..<(end - start) {
                // Exact fp32 head dot (small width, no quantization error).
                let headDot = projectedHeadDot(
                    queryHead: queryHead,
                    rowProjection: proj + offset * headWidth,
                    headWidth: headWidth
                )

                // Tail: x_t·y_t ≤ ‖x_t‖ · tailNorm (sign-safe inflation).
                let rowTail = tailNormPlane[offset]
                let tailUpper = context.tailNorm * rowTail * (1 + slack) + absoluteSlack

                let meanTerm = context.dotMean + vDotMuPlane[offset]
                let inflation = slack
                    * (Swift.abs(headDot) + Swift.abs(meanTerm)
                        + Swift.abs(meanNormSq) + Swift.abs(tailUpper))
                let dotUpper = headDot * (1 + slack) + tailUpper + meanTerm
                    - meanNormSq + inflation + absoluteSlack

                let rowNormSq = rowNormSqPlane[offset]

                switch metric {
                case .cosine:
                    if rowNormSq < 1e-20 || context.norm < 1e-10 {
                        out[offset] = 1.0
                        continue
                    }
                    let denominator = context.norm * sqrt(rowNormSq)
                    var simUpperBound = (dotUpper - absoluteSlack) / denominator
                    if simUpperBound.isNaN || simUpperBound > 1 { simUpperBound = 1 }
                    out[offset] = 1.0 - simUpperBound
                case .innerProduct:
                    // minimized score = −q·v ≥ −dotUpper.
                    out[offset] = -dotUpper
                case .l2:
                    // d² = ‖q‖² − 2q·v + ‖v‖² ≥ ‖q‖² − 2·dotUpper + ‖v‖²(1−δ).
                    var lowerNormSq = rowNormSq * (1 - slack) - absoluteSlack
                    if lowerNormSq < 0 { lowerNormSq = 0 }
                    var gap = context.normSq - 2 * dotUpper + lowerNormSq
                    if gap < 0 { gap = 0 }
                    out[offset] = gap
                case .hamming:
                    break
                }
            }
        }
    }

    @inline(__always)
    private static func projectedHeadDot(
        queryHead: [Float],
        rowProjection: UnsafePointer<Float>,
        headWidth: Int
    ) -> Float {
        var headDot: Float = 0
        var column = 0
        if headWidth >= 4 {
            var vector0 = simd_float4(repeating: 0)
            var vector1 = simd_float4(repeating: 0)
            var vector2 = simd_float4(repeating: 0)
            var vector3 = simd_float4(repeating: 0)
            while column + 15 < headWidth {
                vector0 += simd_float4(
                    queryHead[column], queryHead[column + 1],
                    queryHead[column + 2], queryHead[column + 3]
                ) * simd_float4(
                    rowProjection[column], rowProjection[column + 1],
                    rowProjection[column + 2], rowProjection[column + 3]
                )
                vector1 += simd_float4(
                    queryHead[column + 4], queryHead[column + 5],
                    queryHead[column + 6], queryHead[column + 7]
                ) * simd_float4(
                    rowProjection[column + 4], rowProjection[column + 5],
                    rowProjection[column + 6], rowProjection[column + 7]
                )
                vector2 += simd_float4(
                    queryHead[column + 8], queryHead[column + 9],
                    queryHead[column + 10], queryHead[column + 11]
                ) * simd_float4(
                    rowProjection[column + 8], rowProjection[column + 9],
                    rowProjection[column + 10], rowProjection[column + 11]
                )
                vector3 += simd_float4(
                    queryHead[column + 12], queryHead[column + 13],
                    queryHead[column + 14], queryHead[column + 15]
                ) * simd_float4(
                    rowProjection[column + 12], rowProjection[column + 13],
                    rowProjection[column + 14], rowProjection[column + 15]
                )
                column += 16
            }
            vector0 += vector1 + vector2 + vector3
            headDot = vector0.x + vector0.y + vector0.z + vector0.w
            while column + 3 < headWidth {
                headDot += simd_dot(
                    simd_float4(
                        queryHead[column], queryHead[column + 1],
                        queryHead[column + 2], queryHead[column + 3]
                    ),
                    simd_float4(
                        rowProjection[column], rowProjection[column + 1],
                        rowProjection[column + 2], rowProjection[column + 3]
                    )
                )
                column += 4
            }
        }
        while column < headWidth {
            headDot += queryHead[column] * rowProjection[column]
            column += 1
        }
        return headDot
    }

    // MARK: - Exact rescoring (CPU float4, mirrors flat_scan_distances order)

    static func exactRescore(
        ids: [UInt32],
        corpus: UnsafePointer<Float>,
        dimensionCount: Int,
        query: [Float],
        queryNormSq: Float,
        metric: Metric,
        into results: inout [(distance: Float, id: UInt32)]
    ) {
        guard !ids.isEmpty else { return }
        let distances = UnsafeMutableBufferPointer<Float>.allocate(capacity: ids.count)
        defer { distances.deallocate() }
        let workers = ProcessInfo.processInfo.activeProcessorCount
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let start = (ids.count * worker) / workers
            let end = (ids.count * (worker + 1)) / workers
            guard start < end else { return }
            query.withUnsafeBufferPointer { queryBuf in
                let queryBase = queryBuf.baseAddress!
                for slot in start..<end {
                    let rowBase = corpus + Int(ids[slot]) * dimensionCount
                    var dotQV: Float = 0
                    var normVSq: Float = 0
                    if dimensionCount % 4 == 0 {
                        var offset = 0
                        while offset < dimensionCount {
                            let queryVec = simd_float4(
                                queryBase[offset], queryBase[offset + 1],
                                queryBase[offset + 2], queryBase[offset + 3]
                            )
                            let rowVec = simd_float4(
                                rowBase[offset], rowBase[offset + 1],
                                rowBase[offset + 2], rowBase[offset + 3]
                            )
                            dotQV += simd_dot(queryVec, rowVec)
                            normVSq += simd_dot(rowVec, rowVec)
                            offset += 4
                        }
                    } else {
                        for offset in 0..<dimensionCount {
                            let queryValue = queryBase[offset]
                            let rowValue = rowBase[offset]
                            dotQV += queryValue * rowValue
                            normVSq += rowValue * rowValue
                        }
                    }
                    distances[slot] = finalizeScore(
                        dotQV: dotQV, normVSq: normVSq,
                        metric: metric, queryNormSq: queryNormSq
                    )
                }
            }
        }
        for slot in 0..<ids.count {
            results.append((distances[slot], ids[slot]))
        }
    }

    /// Identical formulas to FlatSearch.metal's flat_finalize_metric.
    @inline(__always)
    static func finalizeScore(
        dotQV: Float, normVSq: Float, metric: Metric, queryNormSq: Float
    ) -> Float {
        switch metric {
        case .l2:
            return Swift.max(0.0, queryNormSq - 2.0 * dotQV + normVSq)
        case .innerProduct:
            return -dotQV
        default:
            let denom = sqrt(queryNormSq) * sqrt(normVSq)
            return denom < 1e-10 ? 1.0 : (1.0 - dotQV / denom)
        }
    }

    static func lowestIdsResult(count: Int, take: Int) -> [SearchResult] {
        (0..<Swift.min(take, count)).map { index in
            SearchResult(id: "", score: 1.0, internalID: UInt32(index))
        }
    }
}
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var mixed = state
        mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
        mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
        return mixed ^ (mixed >> 31)
    }
}
