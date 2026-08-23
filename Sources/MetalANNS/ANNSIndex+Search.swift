import Foundation
import MetalANNSCore

extension GraphIndex {
    public func search(
        query: [Float],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard k > 0 else {
            return []
        }

        let searchMetric = metric ?? configuration.metric
        if searchMetric == .hamming, !configuration.useBinary {
            throw ANNSError.searchFailed("metric .hamming requires a binary index")
        }
        let normalizedQuery =
            (searchMetric == .hamming && configuration.useBinary)
            ? Self.quantizeForHamming(query)
            : query
        let deletedCount = softDeletion.deletedCount
        let effectiveK: Int
        if filter != nil {
            effectiveK = min(vectors.count, k * 4 + deletedCount)
        } else {
            effectiveK = min(vectors.count, k + deletedCount)
        }
        let effectiveEf = max(configuration.efSearch, effectiveK)

        let request = SearchRequest(
            query: normalizedQuery,
            resultLimit: k,
            metric: searchMetric,
            filter: filter,
            maxDistance: nil,
            fetchK: effectiveK,
            fetchEf: effectiveEf,
            context: context,
            vectors: vectors,
            graph: graph,
            entryPoint: entryPoint,
            degree: configuration.degree,
            exactSearchMaxVectorCount: configuration.exactSearchMaxVectorCount,
            softDeletion: softDeletion,
            metadataStore: metadataStore,
            idMap: idMap
        )

        return try await GraphSearchEngine.run(
            request,
            tiers: [FlatExactSearchTier(), HybridGPUBeamSearchTier()],
            prepareCPU: { try await self.prepareCPULadderState(vectors: vectors, graph: graph) },
            metrics: metrics
        )
    }

    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard maxDistance >= 0 else {
            return []
        }
        guard limit > 0 else {
            return []
        }

        let searchMetric = metric ?? configuration.metric
        if searchMetric == .hamming, !configuration.useBinary {
            throw ANNSError.searchFailed("metric .hamming requires a binary index")
        }
        let normalizedQuery =
            (searchMetric == .hamming && configuration.useBinary)
            ? Self.quantizeForHamming(query)
            : query
        let deletedCount = softDeletion.deletedCount
        let searchK = min(vectors.count, limit + deletedCount)
        let searchEf = min(vectors.count, max(configuration.efSearch, searchK * 2))

        let request = SearchRequest(
            query: normalizedQuery,
            resultLimit: limit,
            metric: searchMetric,
            filter: filter,
            maxDistance: maxDistance,
            fetchK: searchK,
            fetchEf: searchEf,
            context: context,
            vectors: vectors,
            graph: graph,
            entryPoint: entryPoint,
            degree: configuration.degree,
            exactSearchMaxVectorCount: configuration.exactSearchMaxVectorCount,
            softDeletion: softDeletion,
            metadataStore: metadataStore,
            idMap: idMap
        )

        return try await GraphSearchEngine.run(
            request,
            tiers: [HybridGPUBeamSearchTier()],
            prepareCPU: { try await self.prepareCPULadderState(vectors: vectors, graph: graph) },
            metrics: metrics
        )
    }

    public func batchSearch(
        queries: [[Float]],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [[SearchResult]] {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard !queries.isEmpty else {
            return []
        }
        if let metrics {
            await metrics.recordBatchSearch()
        }

        // Exact even with soft deletions: fetch top-(k+deleted) candidates,
        // then drop deleted rows — each deletion demotes a survivor by at most
        // one rank, so the filtered list is the true top-k survivors.
        let deletedCount = softDeletion.deletedCount
        let batchEffectiveK = min((vectors?.count ?? k), k + deletedCount)
        if let vectors, filter == nil,
            MetalANNSCore.FlatGPUSearch.isEligible(
                vectors: vectors,
                metric: metric ?? configuration.metric,
                k: batchEffectiveK,
                maxVectorCount: configuration.exactSearchMaxVectorCount
            )
        {
            do {
                let flatResults = try await MetalANNSCore.FlatGPUSearch.batchSearch(
                    context: context,
                    queries: queries,
                    vectors: vectors,
                    k: batchEffectiveK,
                    metric: metric ?? configuration.metric
                )
                let mapped = flatResults.map { results in
                    GraphSearchEngine.postProcess(
                        results,
                        softDeletion: softDeletion,
                        filter: nil,
                        metadataStore: metadataStore,
                        idMap: idMap,
                        maxDistance: nil,
                        limit: k
                    )
                }
                return mapped
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // Fall through to the per-query path below.
            }
        }

        let maxConcurrency = await batchSearchMaxConcurrency()

        return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
            var orderedResults = [[SearchResult]?](repeating: nil, count: queries.count)
            var nextIndex = 0

            for _ in 0..<min(maxConcurrency, queries.count) {
                let idx = nextIndex
                let query = queries[idx]
                nextIndex += 1
                group.addTask { [self] in
                    let result = try await self.search(query: query, k: k, filter: filter, metric: metric)
                    return (idx, result)
                }
            }

            for try await (idx, result) in group {
                orderedResults[idx] = result
                if nextIndex < queries.count {
                    let idx = nextIndex
                    let query = queries[idx]
                    nextIndex += 1
                    group.addTask { [self] in
                        let result = try await self.search(query: query, k: k, filter: filter, metric: metric)
                        return (idx, result)
                    }
                }
            }

            return orderedResults.map { $0! }
        }
    }

    /// Prepares CPU traversal inputs for the engine's fallback ladder. Runs on
    /// the actor so the lazy HNSW rebuild stays coordinated with other state.
    func prepareCPULadderState(vectors: any VectorStorage, graph: GraphBuffer) throws -> PreparedCPUSearch {
        if configuration.hnswConfiguration.enabled, hnsw == nil {
            try rebuildHNSWFromCurrentState()
        }
        return PreparedCPUSearch(
            vectors: extractVectors(from: vectors),
            graph: extractGraph(from: graph),
            hnsw: hnsw,
            baseMetric: configuration.metric
        )
    }

    public func setMetrics(_ metrics: IndexMetrics?) {
        self.metrics = metrics
    }

    public func batchSearchMaxConcurrencyForTesting() async -> Int {
        await batchSearchMaxConcurrency()
    }

}
