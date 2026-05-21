import Accelerate
import Foundation

public struct AccelerateBackend: ComputeBackend, Sendable {
    public init() {}

    public func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float] {
        guard query.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }
        let expectedCount = vectorCount * dim
        guard vectors.count == expectedCount else {
            throw ANNSError.dimensionMismatch(expected: expectedCount, got: vectors.count)
        }

        var results = [Float](repeating: 0, count: vectorCount)

        switch metric {
        case .cosine:
            computeCosineDistances(
                query: query,
                vectors: vectors,
                vectorCount: vectorCount,
                dim: dim,
                results: &results
            )
        case .l2:
            computeL2Distances(
                query: query,
                vectors: vectors,
                vectorCount: vectorCount,
                dim: dim,
                results: &results
            )
        case .innerProduct:
            computeInnerProductDistances(
                query: query,
                vectors: vectors,
                vectorCount: vectorCount,
                dim: dim,
                results: &results
            )
        case .hamming:
            computeHammingDistances(
                query: query,
                vectors: vectors,
                vectorCount: vectorCount,
                dim: dim,
                results: &results
            )
        }

        return results
    }

    // FIXME: Blocker — queryBase escapes withUnsafeBufferPointer closure (UB).
    // The pointer is only valid inside the closure; using it in the loop below
    // is undefined behavior. Restructure to keep pointer usage inside the
    // closure or use withUnsafeBytes for the entire computation.
    private func computeCosineDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        results: inout [Float]
    ) {
        guard
            let vectorsBase = vectors.baseAddress,
            let queryBase = query.withUnsafeBufferPointer({ $0.baseAddress })
        else {
            return
        }

        var queryNormSq: Float = 0
        vDSP_dotpr(queryBase, 1, queryBase, 1, &queryNormSq, vDSP_Length(dim))
        let queryNorm = sqrt(queryNormSq)

        for index in 0..<vectorCount {
            let vectorBase = vectorsBase + (index * dim)

            var dot: Float = 0
            vDSP_dotpr(queryBase, 1, vectorBase, 1, &dot, vDSP_Length(dim))

            var vectorNormSq: Float = 0
            vDSP_dotpr(vectorBase, 1, vectorBase, 1, &vectorNormSq, vDSP_Length(dim))
            let vectorNorm = sqrt(vectorNormSq)

            let denom = queryNorm * vectorNorm
            if denom < 1e-10 {
                results[index] = 1.0
            } else {
                results[index] = 1.0 - (dot / denom)
            }
        }
    }

    private func computeL2Distances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        results: inout [Float]
    ) {
        guard let vectorsBase = vectors.baseAddress else {
            return
        }

        for index in 0..<vectorCount {
            let vectorBase = vectorsBase + (index * dim)
            var sumSq: Float = 0
            for d in 0..<dim {
                let diff = query[d] - vectorBase[d]
                sumSq += diff * diff
            }
            results[index] = sumSq
        }
    }

    // FIXME: Blocker — Same dangling-pointer issue as computeCosineDistances.
    // queryBase escapes withUnsafeBufferPointer; restructure before production.
    private func computeInnerProductDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        results: inout [Float]
    ) {
        guard
            let vectorsBase = vectors.baseAddress,
            let queryBase = query.withUnsafeBufferPointer({ $0.baseAddress })
        else {
            return
        }

        for index in 0..<vectorCount {
            let vectorBase = vectorsBase + (index * dim)
            var dot: Float = 0
            vDSP_dotpr(queryBase, 1, vectorBase, 1, &dot, vDSP_Length(dim))
            results[index] = -dot
        }
    }

    private func computeHammingDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        results: inout [Float]
    ) {
        guard let vectorsBase = vectors.baseAddress else {
            return
        }

        for index in 0..<vectorCount {
            let vectorBase = vectorsBase + (index * dim)
            var mismatches = 0
            for d in 0..<dim where query[d] != vectorBase[d] {
                mismatches += 1
            }
            results[index] = Float(mismatches)
        }
    }
}
