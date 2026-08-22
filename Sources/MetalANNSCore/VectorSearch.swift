import Foundation

/// Which adapter executes a vector search.
public enum SearchPath: Sendable {
    /// Heuristic adapter selection; falls back freely when an attempt fails.
    case auto
    /// Exact full-corpus scan (recall 1.0). Strict: throws when unavailable.
    case exact
    /// GPU graph beam search. Strict: throws when unavailable.
    case gpu
    /// CPU graph beam search (HNSW overlay when available, plain beam otherwise).
    case cpu
}

/// Executes a vector search over exactly one adapter, chosen by `SearchPath`.
///
/// This is the single seam between index state and search adapters: workload
/// gating thresholds, try-GPU-catch-CPU fallback, HNSW materialization, and
/// strict-path availability rules all live here — once.
public enum VectorSearch {
    /// A ranked vector-search output: an internal slot and its distance,
    /// before deletion filtering, metadata filtering, and ID mapping.
    public struct Candidate: Sendable {
        public let internalID: UInt32
        public let score: Float

        init(internalID: UInt32, score: Float) {
            self.internalID = internalID
            self.score = score
        }
    }

    // MARK: - Workload gates (.auto only)

    /// GPU beam-search kernel caps k/ef at this bound; larger requests run on CPU.
    static let maxGPUBeamEF = 256
    /// Below this corpus size the GPU dispatch round trip dominates; CPU wins.
    static let minHybridGPUSearchNodeCount = 4_096
    /// Below this estimated distance work the GPU submission tax is not amortized.
    static let minHybridGPUSearchWork = 16_384

    /// Runs one vector search through the requested path.
    ///
    /// - Parameters:
    ///   - allowsExact: Caller-side statement that this query's post-processing
    ///     permits skipping filtered results (no metadata filter, no deletions).
    ///     Gates both `.auto` exact-tier eligibility and strict `.exact`.
    ///   - ensureHNSW: Lazily builds the HNSW overlay before CPU fallback;
    ///     memoization stays with the caller's mutable state.
    public static func execute(
        context: MetalContext?,
        query: [Float],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric,
        degree: Int,
        isBinary: Bool,
        baseMetric: Metric,
        hnswEnabled: Bool,
        hnsw: HNSWLayers?,
        ensureHNSW: @escaping @Sendable () async throws -> HNSWLayers?,
        exactMaxVectorCount: Int,
        allowsExact: Bool,
        path: SearchPath = .auto
    ) async throws -> [Candidate] {
        switch path {
        case .exact:
            guard allowsExact else {
                throw ANNSError.searchFailed(
                    "exact path unavailable: query applies a filter or the index has deleted vectors"
                )
            }
            guard FlatGPUSearch.isEligible(
                vectors: vectors,
                metric: metric,
                k: k,
                maxVectorCount: exactMaxVectorCount
            ) else {
                throw ANNSError.searchFailed(
                    "exact path not eligible for this storage/metric/workload; use .auto, .gpu or .cpu"
                )
            }
            return try await FlatGPUSearch.search(
                context: context,
                query: query,
                vectors: vectors,
                k: k,
                metric: metric
            )
            .map { Candidate($0) }

        case .gpu:
            guard !isBinary else {
                throw ANNSError.searchFailed("GPU graph search does not support binary storage")
            }
            guard metric != .hamming else {
                throw ANNSError.searchFailed("GPU graph search does not support metric .hamming")
            }
            guard k <= maxGPUBeamEF, ef <= maxGPUBeamEF else {
                throw ANNSError.searchFailed(
                    "k/ef exceed the GPU beam cap (\(maxGPUBeamEF)); use .cpu or lower ef"
                )
            }
            guard let context else {
                throw ANNSError.searchFailed("GPU search requires a Metal context")
            }
            return try await graphCandidates(
                context: context,
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: metric
            )

        case .cpu:
            return try await cpuCandidates(
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: metric,
                baseMetric: baseMetric,
                hnswEnabled: hnswEnabled,
                hnsw: hnsw,
                ensureHNSW: ensureHNSW
            )

        case .auto:
            if allowsExact,
               FlatGPUSearch.isEligible(
                   vectors: vectors,
                   metric: metric,
                   k: k,
                   maxVectorCount: exactMaxVectorCount
               ),
               let flat = try? await FlatGPUSearch.search(
                   context: context,
                   query: query,
                   vectors: vectors,
                   k: k,
                   metric: metric
               ) {
                return flat.map { Candidate($0) }
            }

            if shouldUseHybridGPU(
                nodeCount: vectors.count,
                metric: metric,
                isBinary: isBinary,
                degree: degree,
                k: k,
                ef: ef
            ), let context {
                do {
                    return try await graphCandidates(
                        context: context,
                        query: query,
                        vectors: vectors,
                        graph: graph,
                        entryPoint: entryPoint,
                        k: k,
                        ef: ef,
                        metric: metric
                    )
                } catch {
                    return try await cpuCandidates(
                        query: query,
                        vectors: vectors,
                        graph: graph,
                        entryPoint: entryPoint,
                        k: k,
                        ef: ef,
                        metric: metric,
                        baseMetric: baseMetric,
                        hnswEnabled: hnswEnabled,
                        hnsw: hnsw,
                        ensureHNSW: ensureHNSW
                    )
                }
            }

            return try await cpuCandidates(
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: metric,
                baseMetric: baseMetric,
                hnswEnabled: hnswEnabled,
                hnsw: hnsw,
                ensureHNSW: ensureHNSW
            )
        }
    }

    // MARK: - Adapters

    private static func graphCandidates(
        context: MetalContext,
        query: [Float],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [Candidate] {
        try await SearchGPU.search(
            context: context,
            query: query,
            vectors: vectors,
            graph: graph,
            entryPoint: entryPoint,
            k: max(1, k),
            ef: max(1, ef),
            metric: metric
        )
        .map { Candidate($0) }
    }

    private static func cpuCandidates(
        query: [Float],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric,
        baseMetric: Metric,
        hnswEnabled: Bool,
        hnsw: HNSWLayers?,
        ensureHNSW: () async throws -> HNSWLayers?
    ) async throws -> [Candidate] {
        var effectiveHNSW = hnsw
        if hnswEnabled, effectiveHNSW == nil {
            effectiveHNSW = try await ensureHNSW()
        }

        if let effectiveHNSW, metric == baseMetric {
            return try await HNSWSearchCPU.search(
                query: query,
                vectors: extractVectors(from: vectors),
                hnsw: effectiveHNSW,
                baseGraph: extractGraph(from: graph),
                k: max(1, k),
                ef: max(1, ef),
                metric: metric
            )
            .map { Candidate($0) }
        }

        return try await BeamSearchCPU.search(
            query: query,
            vectors: extractVectors(from: vectors),
            graph: extractGraph(from: graph),
            entryPoint: entryPoint,
            k: max(1, k),
            ef: max(1, ef),
            metric: metric
        )
        .map { Candidate($0) }
    }

    private static func shouldUseHybridGPU(
        nodeCount: Int,
        metric: Metric,
        isBinary: Bool,
        degree: Int,
        k: Int,
        ef: Int
    ) -> Bool {
        guard !isBinary else { return false }
        guard metric != .hamming else { return false }
        guard k <= maxGPUBeamEF, ef <= maxGPUBeamEF else { return false }
        guard nodeCount >= minHybridGPUSearchNodeCount else { return false }
        let estimatedDistanceWork = max(k, ef) * max(degree, 1)
        return estimatedDistanceWork >= minHybridGPUSearchWork
    }

    // MARK: - Materialization helpers

    /// Copies buffer-backed vectors into row-major memory for CPU adapters.
    package static func extractVectors(from vectors: any VectorStorage) -> [[Float]] {
        (0..<vectors.count).map { vectors.vector(at: $0) }
    }

    /// Copies a GPU graph buffer into adjacency-list form for CPU adapters.
    package static func extractGraph(from graph: GraphBuffer) -> [[(UInt32, Float)]] {
        (0..<graph.nodeCount).map { nodeID in
            let ids = graph.neighborIDs(of: nodeID)
            let distances = graph.neighborDistances(of: nodeID)
            return zip(ids, distances)
                .filter { $0.0 != UInt32.max }
                .map { ($0.0, $0.1) }
        }
    }
}

extension VectorSearch.Candidate {
    init(_ result: SearchResult) {
        self.init(internalID: result.internalID, score: result.score)
    }
}
