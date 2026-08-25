import Foundation
import Metal

package enum FullGPUSearch {
    private static let maxEF = 256

    package static func search(
        context: MetalContext,
        query: [Float],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        let nodeCount =
            graph.nodeCount > 0
            ? min(graph.nodeCount, vectors.count)
            : min(graph.capacity, vectors.count)

        guard nodeCount > 0 else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }
        guard ef >= k else {
            throw ANNSError.searchFailed("ef must be greater than or equal to k")
        }
        guard k <= maxEF else {
            throw ANNSError.searchFailed("k exceeds FullGPUSearch maximum (\(maxEF)); use CPU/hybrid search")
        }
        guard ef <= maxEF else {
            throw ANNSError.searchFailed("ef exceeds FullGPUSearch maximum (\(maxEF)); use CPU/hybrid search")
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard entryPoint >= 0, entryPoint < nodeCount else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        let kLimit = min(k, nodeCount)
        let efLimit = min(max(ef, kLimit), nodeCount)
        guard metric != .hamming else {
            throw ANNSError.searchFailed("FullGPUSearch does not support metric .hamming")
        }

        // Full GPU beam-search kernel currently exhibits recall regressions on
        // larger/random graphs. Delegate to the hybrid GPU distance path for
        // correctness parity while preserving GPU acceleration for distance math.
        return try await SearchGPU.search(
            context: context,
            query: query,
            vectors: vectors,
            graph: graph,
            entryPoint: entryPoint,
            k: kLimit,
            ef: efLimit,
            metric: metric
        )
    }
}
