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
        let metricsRecorder = metrics
        let searchStart = metricsRecorder == nil ? nil : ContinuousClock.now
        let hasFilter = filter != nil
        let deletedCount = softDeletion.deletedCount
        let effectiveK: Int
        if hasFilter {
            effectiveK = min(vectors.count, k * 4 + deletedCount)
        } else {
            effectiveK = min(vectors.count, k + deletedCount)
        }
        let effectiveEf = max(configuration.efSearch, effectiveK)

        var pendingFlatResults: [SearchResult]? = nil
        if filter == nil,
            // Exact even with soft deletions: scanning top-(k+deletedCount)
            // and filtering deleted rows afterwards yields the true top-k
            // survivors, since each deleted row demotes a survivor by at most
            // one position.
            MetalANNSCore.FlatGPUSearch.isEligible(
                vectors: vectors,
                metric: searchMetric,
                k: effectiveK,
                maxVectorCount: configuration.exactSearchMaxVectorCount
            )
        {
            do {
                pendingFlatResults = try await MetalANNSCore.FlatGPUSearch.search(
                    context: context,
                    query: normalizedQuery,
                    vectors: vectors,
                    k: effectiveK,
                    metric: searchMetric
                )
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // Fall back to the graph traversal backend below.
                pendingFlatResults = nil
            }
        }

        let rawResults: [SearchResult]
        if let flatResults = pendingFlatResults {
            rawResults = flatResults
        } else if let context,
            shouldUseHybridGPUSearch(
                for: vectors,
                metric: searchMetric,
                k: effectiveK,
                ef: effectiveEf
            )
        {
            do {
                rawResults = try await SearchGPU.search(
                    context: context,
                    query: normalizedQuery,
                    vectors: vectors,
                    graph: graph,
                    entryPoint: Int(entryPoint),
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            } catch {
                if configuration.hnswConfiguration.enabled, hnsw == nil {
                    try rebuildHNSWFromCurrentState()
                }

                let extractedVectors = extractVectors(from: vectors)
                let extractedGraph = extractGraph(from: graph)

                if let hnsw, searchMetric == configuration.metric {
                    rawResults = try await HNSWSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        hnsw: hnsw,
                        baseGraph: extractedGraph,
                        k: max(1, effectiveK),
                        ef: max(1, effectiveEf),
                        metric: searchMetric
                    )
                } else {
                    rawResults = try await BeamSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        graph: extractedGraph,
                        entryPoint: Int(entryPoint),
                        k: max(1, effectiveK),
                        ef: max(1, effectiveEf),
                        metric: searchMetric
                    )
                }
            }
        } else {
            if configuration.hnswConfiguration.enabled, hnsw == nil {
                try rebuildHNSWFromCurrentState()
            }

            let extractedVectors = extractVectors(from: vectors)
            let extractedGraph = extractGraph(from: graph)

            if let hnsw, searchMetric == configuration.metric {
                rawResults = try await HNSWSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    hnsw: hnsw,
                    baseGraph: extractedGraph,
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            } else {
                rawResults = try await BeamSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    graph: extractedGraph,
                    entryPoint: Int(entryPoint),
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            }
        }

        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
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
        let output = Array(mapped.prefix(k))
        if let metricsRecorder, let searchStart {
            let duration = ContinuousClock.now - searchStart
            await metricsRecorder.recordSearch(durationNs: Self.durationNanoseconds(duration))
        }
        return output
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
        let metricsRecorder = metrics
        let searchStart = metricsRecorder == nil ? nil : ContinuousClock.now
        let deletedCount = softDeletion.deletedCount
        let searchK = min(vectors.count, limit + deletedCount)
        let searchEf = min(vectors.count, max(configuration.efSearch, searchK * 2))

        let rawResults: [SearchResult]
        let canAttemptGPU = shouldUseHybridGPUSearch(
            for: vectors,
            metric: searchMetric,
            k: searchK,
            ef: searchEf
        )

        if let context, canAttemptGPU {
            do {
                rawResults = try await SearchGPU.search(
                    context: context,
                    query: normalizedQuery,
                    vectors: vectors,
                    graph: graph,
                    entryPoint: Int(entryPoint),
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            } catch {
                if configuration.hnswConfiguration.enabled, hnsw == nil {
                    try rebuildHNSWFromCurrentState()
                }

                let extractedVectors = extractVectors(from: vectors)
                let extractedGraph = extractGraph(from: graph)

                if let hnsw, searchMetric == configuration.metric {
                    rawResults = try await HNSWSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        hnsw: hnsw,
                        baseGraph: extractedGraph,
                        k: max(1, searchK),
                        ef: max(1, searchEf),
                        metric: searchMetric
                    )
                } else {
                    rawResults = try await BeamSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        graph: extractedGraph,
                        entryPoint: Int(entryPoint),
                        k: max(1, searchK),
                        ef: max(1, searchEf),
                        metric: searchMetric
                    )
                }
            }
        } else {
            if configuration.hnswConfiguration.enabled, hnsw == nil {
                try rebuildHNSWFromCurrentState()
            }

            let extractedVectors = extractVectors(from: vectors)
            let extractedGraph = extractGraph(from: graph)

            if let hnsw, searchMetric == configuration.metric {
                rawResults = try await HNSWSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    hnsw: hnsw,
                    baseGraph: extractedGraph,
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            } else {
                rawResults = try await BeamSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    graph: extractedGraph,
                    entryPoint: Int(entryPoint),
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            }
        }

        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
        }
        let withinRange = filtered.filter { $0.score <= maxDistance }

        let mapped = withinRange.compactMap { result -> SearchResult? in
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
        let output = Array(mapped.prefix(limit))
        if let metricsRecorder, let searchStart {
            let duration = ContinuousClock.now - searchStart
            await metricsRecorder.recordSearch(durationNs: Self.durationNanoseconds(duration))
        }
        return output
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
                    softDeletion.filterResults(results).compactMap { result -> SearchResult? in
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
                }
                return mapped.map { Array($0.prefix(k)) }
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

    public func setMetrics(_ metrics: IndexMetrics?) {
        self.metrics = metrics
    }

    public func batchSearchMaxConcurrencyForTesting() async -> Int {
        await batchSearchMaxConcurrency()
    }

}
