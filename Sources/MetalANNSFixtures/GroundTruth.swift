import Foundation
import MetalANNSCore

extension Fixtures {
    /// Brute-force nearest neighbors of `query` among `vectors`, returned as
    /// internal ids ordered nearest-first. `k` is clamped to
    /// `1...vectors.count`, matching the historical benchmark copies.
    public static func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> [UInt32] {
        let topK = max(1, min(k, vectors.count))
        return
            vectors
            .enumerated()
            .map { idx, vector -> (id: UInt32, distance: Float) in
                (UInt32(idx), distance(query: query, vector: vector, metric: metric))
            }
            .sorted { $0.distance < $1.distance }
            .prefix(topK)
            .map(\.id)
    }

    /// Brute-force top-k as a set with deterministic index tie-breaking and
    /// optional row exclusion, matching the exactness-regression test flavor.
    public static func exactTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric,
        skipIndices: Set<Int> = []
    ) -> Set<UInt32> {
        let scored = vectors.enumerated().compactMap { index, vector -> (UInt32, Float)? in
            guard !skipIndices.contains(index) else { return nil }
            return (UInt32(index), distance(query: query, vector: vector, metric: metric))
        }
        return Set(
            scored.sorted { lhs, rhs in
                lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
            }
            .prefix(k)
            .map(\.0))
    }

    /// Reference CPU distance used for ground-truth computation.
    public static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        case .hamming:
            var mismatches = 0
            for d in 0..<query.count where query[d] != vector[d] {
                mismatches += 1
            }
            return Float(mismatches)
        }
    }
}
