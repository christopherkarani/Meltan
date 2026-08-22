import Foundation
import MetalANNSCore

/// Everything one graph-index search needs, validated once by the entry-point
/// adapter and passed through the engine as an immutable snapshot.
struct SearchRequest: Sendable {
    let query: [Float]
    let resultLimit: Int
    let metric: Metric
    let filter: _LegacySearchFilter?
    let maxDistance: Float?
    let fetchK: Int
    let fetchEf: Int
    let context: MetalContext?
    let vectors: any VectorStorage
    let graph: GraphBuffer
    let entryPoint: UInt32
    let degree: Int
    let exactSearchMaxVectorCount: Int
    let baseMetric: Metric
    let softDeletion: SoftDeletion
    let metadataStore: MetadataStore
    let idMap: IDMap
}

/// CPU traversal inputs prepared on the `GraphIndex` actor (lazy HNSW rebuild
/// plus vector/graph extraction) and handed to the engine's fallback ladder.
struct PreparedCPUSearch: Sendable {
    let vectors: [[Float]]
    let graph: [[(UInt32, Float)]]
    let hnsw: HNSWLayers?
    let baseMetric: Metric
}

/// One ordered step of the search cascade; each tier knows its own eligibility.
protocol SearchTier: Sendable {
    func isEligible(_ request: SearchRequest) -> Bool

    /// Returns the tier's raw results, or nil when it cannot produce any and
    /// the ladder should continue. Thrown `CancellationError` always aborts.
    func execute(_ request: SearchRequest) async throws -> [SearchResult]?
}

/// Exact flat scan (GPU or host BLAS).
///
/// Eligible only without filters: scanning top-(k+deletedCount) and filtering
/// deleted rows afterwards yields the true top-k survivors, since each deleted
/// row demotes a survivor by at most one position. Metadata filters break that
/// argument, so filtered workloads skip this tier entirely.
struct FlatExactSearchTier: SearchTier {
    func isEligible(_ request: SearchRequest) -> Bool {
        guard request.filter == nil else {
            return false
        }
        return MetalANNSCore.FlatGPUSearch.isEligible(
            vectors: request.vectors,
            metric: request.metric,
            k: request.fetchK,
            maxVectorCount: request.exactSearchMaxVectorCount
        )
    }

    func execute(_ request: SearchRequest) async throws -> [SearchResult]? {
        try await MetalANNSCore.FlatGPUSearch.search(
            context: request.context,
            query: request.query,
            vectors: request.vectors,
            k: request.fetchK,
            metric: request.metric
        )
    }
}

/// Hybrid GPU-assisted beam traversal over the CAGRA graph.
struct HybridGPUBeamSearchTier: SearchTier {
    func isEligible(_ request: SearchRequest) -> Bool {
        guard request.context != nil else {
            return false
        }
        return Self.shouldUseHybridGPU(
            vectors: request.vectors,
            metric: request.metric,
            k: request.fetchK,
            ef: request.fetchEf,
            degree: request.degree
        )
    }

    func execute(_ request: SearchRequest) async throws -> [SearchResult]? {
        guard let context = request.context else {
            return nil
        }
        return try await SearchGPU.search(
            context: context,
            query: request.query,
            vectors: request.vectors,
            graph: request.graph,
            entryPoint: Int(request.entryPoint),
            k: max(1, request.fetchK),
            ef: max(1, request.fetchEf),
            metric: request.metric
        )
    }

    /// Workload gating moved verbatim from `GraphIndex.shouldUseHybridGPUSearch`
    /// (threshold constants stay on `GraphIndex`).
    static func shouldUseHybridGPU(
        vectors: any VectorStorage,
        metric: Metric,
        k: Int,
        ef: Int,
        degree: Int
    ) -> Bool {
        guard !(vectors is BinaryVectorBuffer) else {
            return false
        }
        guard metric != .hamming else {
            return false
        }
        guard k <= GraphIndex.fullGPUMaxEF, ef <= GraphIndex.fullGPUMaxEF else {
            return false
        }

        let nodeCount = vectors.count
        guard nodeCount >= GraphIndex.minHybridGPUSearchNodeCount else {
            return false
        }

        let estimatedDistanceWork = max(k, ef) * max(degree, 1)
        return estimatedDistanceWork >= GraphIndex.minHybridGPUSearchWork
    }
}

/// Single execution seam for the graph-index search cascade: owns the ordered
/// tier ladder, the cancellation policy, one-shot CPU fallback accounting,
/// post-processing, and search-metrics timing.
enum GraphSearchEngine {
    /// Runs the cascade for one request: first eligible tier wins, generic
    /// failures degrade to the next tier, `CancellationError` rethrows
    /// unwrapped from every tier, and when no tier produces results the
    /// prepared CPU ladder runs exactly once.
    static func run(
        _ request: SearchRequest,
        tiers: [any SearchTier],
        prepareCPU: @Sendable () async throws -> PreparedCPUSearch,
        metrics: IndexMetrics?
    ) async throws -> [SearchResult] {
        let searchStart = metrics == nil ? nil : ContinuousClock.now

        var pendingResults: [SearchResult]? = nil
        for tier in tiers where tier.isEligible(request) {
            do {
                if let results = try await tier.execute(request) {
                    pendingResults = results
                    break
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                continue
            }
        }

        let rawResults: [SearchResult]
        if let pendingResults {
            rawResults = pendingResults
        } else {
            let prepared = try await prepareCPU()
            rawResults = try await executeCPULadder(prepared, request: request)
        }

        let output = postProcess(
            rawResults,
            softDeletion: request.softDeletion,
            filter: request.filter,
            metadataStore: request.metadataStore,
            idMap: request.idMap,
            maxDistance: request.maxDistance,
            limit: request.resultLimit
        )

        if let metrics, let searchStart {
            let duration = ContinuousClock.now - searchStart
            await metrics.recordSearch(durationNs: GraphIndex.durationNanoseconds(duration))
        }
        return output
    }

    /// The single HNSW-vs-beam selection and execution site.
    static func executeCPULadder(
        _ prepared: PreparedCPUSearch,
        request: SearchRequest
    ) async throws -> [SearchResult] {
        if let hnsw = prepared.hnsw, request.metric == prepared.baseMetric {
            return try await HNSWSearchCPU.search(
                query: request.query,
                vectors: prepared.vectors,
                hnsw: hnsw,
                baseGraph: prepared.graph,
                k: max(1, request.fetchK),
                ef: max(1, request.fetchEf),
                metric: request.metric
            )
        }
        return try await BeamSearchCPU.search(
            query: request.query,
            vectors: prepared.vectors,
            graph: prepared.graph,
            entryPoint: Int(request.entryPoint),
            k: max(1, request.fetchK),
            ef: max(1, request.fetchEf),
            metric: request.metric
        )
    }

    /// Shared result chain: deletion filter → metadata filter → range cut →
    /// id-map drop-unmapped mapping → prefix(limit).
    static func postProcess(
        _ rawResults: [SearchResult],
        softDeletion: SoftDeletion,
        filter: _LegacySearchFilter?,
        metadataStore: MetadataStore,
        idMap: IDMap,
        maxDistance: Float?,
        limit: Int
    ) -> [SearchResult] {
        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
        }
        if let maxDistance {
            filtered = filtered.filter { $0.score <= maxDistance }
        }

        let mapped = filtered.compactMap { result -> SearchResult? in
            let externalID = idMap.externalID(for: result.internalID) ?? ""
            let numericID = idMap.numericID(for: result.internalID)
            guard !externalID.isEmpty || numericID != nil else {
                return nil
            }
            return SearchResult(
                id: externalID,
                score: result.score,
                internalID: result.internalID,
                numericID: numericID
            )
        }
        return Array(mapped.prefix(limit))
    }
}
