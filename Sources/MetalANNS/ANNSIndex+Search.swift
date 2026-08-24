import Foundation
import MetalANNSCore

extension GraphIndex {
    /// - Parameters:
    ///   - searchMode: Overrides `IndexConfiguration.searchMode` for this call.
    ///     `nil` uses the configuration. `.fast` is approximate IVF-flat.
    ///   - nprobe: Overrides `IndexConfiguration.ivfNProbe` for `.fast`.
    public func search(
        query: [Float],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil,
        searchMode: SearchMode? = nil,
        nprobe: Int? = nil
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
            searchMode: searchMode ?? configuration.searchMode,
            ivfListCount: configuration.ivfListCount,
            ivfNProbe: nprobe ?? configuration.ivfNProbe,
            softDeletion: softDeletion,
            metadataStore: metadataStore,
            idMap: idMap
        )

        return try await GraphSearchEngine.run(
            request,
            tiers: [IVFFlatSearchTier(), FlatExactSearchTier(), HybridGPUBeamSearchTier()],
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
            searchMode: configuration.searchMode,
            ivfListCount: configuration.ivfListCount,
            ivfNProbe: configuration.ivfNProbe,
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
        let batchMetric = metric ?? configuration.metric
        if configuration.searchMode == .fast, filter == nil,
            let vectorBuffer = vectors as? VectorBuffer,
            !vectorBuffer.isFloat16,
            batchMetric != .hamming,
            vectorBuffer.count >= IVFFlatSearch.minVectorCount
        {
            var mapped: [[SearchResult]] = []
            mapped.reserveCapacity(queries.count)
            for query in queries {
                try Task.checkCancellation()
                let raw = try IVFFlatSearch.search(
                    query: query,
                    vectors: vectorBuffer,
                    k: batchEffectiveK,
                    nlist: configuration.ivfListCount,
                    nprobe: configuration.ivfNProbe,
                    metric: batchMetric
                )
                mapped.append(
                    GraphSearchEngine.postProcess(
                        raw,
                        softDeletion: softDeletion,
                        filter: nil,
                        metadataStore: metadataStore,
                        idMap: idMap,
                        maxDistance: nil,
                        limit: k
                    )
                )
            }
            return mapped
        }
        if let vectors, filter == nil,
            MetalANNSCore.FlatGPUSearch.isEligible(
                vectors: vectors,
                metric: batchMetric,
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
                    metric: batchMetric
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
            } catch {
                guard let tierError = error as? ANNSError, tierError.isTierDegradeable else {
                    throw error
                }
                // Fall through to the per-query path below.
            }
        }

        let maxConcurrency = await batchSearchMaxConcurrency()

        return try await BatchExecution.run(
            over: queries,
            maxConcurrency: maxConcurrency
        ) { [self] query in
            try await search(query: query, k: k, filter: filter, metric: metric)
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
