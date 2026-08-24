import Foundation
import MetalANNSCore

public actor _ShardedIndex {
    private let numShards: Int
    private let nprobe: Int
    private let configuration: IndexConfiguration

    private var centroids: [[Float]] = []
    private var shards: [GraphIndex] = []
    private var isBuilt = false

    public init(
        numShards: Int = 16,
        nprobe: Int = 4,
        configuration: IndexConfiguration = .default
    ) {
        let normalizedShards = max(1, numShards)
        self.numShards = normalizedShards
        self.nprobe = max(1, min(nprobe, normalizedShards))
        self.configuration = configuration
    }

    public func build(vectors: [[Float]], ids: [String]) async throws {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build sharded index with empty vectors")
        }
        guard vectors.count > 1 else {
            throw ANNSError.constructionFailed("Sharded index requires at least 2 vectors")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        let effectiveShards = min(numShards, vectors.count)
        let sharedContext = try? MetalContext()

        let kmeans = try KMeans.cluster(
            vectors: vectors,
            k: effectiveShards,
            maxIterations: 20,
            metric: configuration.metric,
            seed: 42
        )

        var shardVectors = Array(repeating: [[Float]](), count: effectiveShards)
        var shardIDs = Array(repeating: [String](), count: effectiveShards)

        for i in 0..<vectors.count {
            let shardIndex = kmeans.assignments[i]
            shardVectors[shardIndex].append(vectors[i])
            shardIDs[shardIndex].append(ids[i])
        }

        let eligibleTargets = shardVectors.indices.filter { shardVectors[$0].count >= 2 }
        if !eligibleTargets.isEmpty {
            for shardIndex in shardVectors.indices where shardVectors[shardIndex].count == 1 {
                let loneVector = shardVectors[shardIndex][0]
                var targetShard = eligibleTargets[0]
                var bestDistance = SIMDDistance.distance(
                    loneVector,
                    kmeans.centroids[targetShard],
                    metric: configuration.metric
                )

                for candidate in eligibleTargets.dropFirst() {
                    let distance = SIMDDistance.distance(
                        loneVector,
                        kmeans.centroids[candidate],
                        metric: configuration.metric
                    )
                    if distance < bestDistance {
                        bestDistance = distance
                        targetShard = candidate
                    }
                }

                shardVectors[targetShard].append(loneVector)
                shardIDs[targetShard].append(shardIDs[shardIndex][0])
                shardVectors[shardIndex].removeAll(keepingCapacity: true)
                shardIDs[shardIndex].removeAll(keepingCapacity: true)
            }
        }

        var indexedShards: [(index: Int, shard: GraphIndex)] = []
        indexedShards.reserveCapacity(effectiveShards)

        try await withThrowingTaskGroup(of: (Int, GraphIndex).self) { group in
            for shardIndex in 0..<effectiveShards {
                guard !shardVectors[shardIndex].isEmpty else {
                    continue
                }

                let shardData = shardVectors[shardIndex]
                let shardDataIDs = shardIDs[shardIndex]
                var shardConfiguration = configuration
                shardConfiguration.degree = min(configuration.degree, max(1, shardData.count - 1))

                group.addTask {
                    let shard = GraphIndex(
                        configuration: shardConfiguration,
                        context: sharedContext
                    )
                    try await shard.build(vectors: shardData, ids: shardDataIDs)
                    return (shardIndex, shard)
                }
            }

            for try await (index, shard) in group {
                indexedShards.append((index, shard))
            }
        }

        indexedShards.sort { $0.index < $1.index }
        let builtShards = indexedShards.map(\.shard)
        let builtCentroids = indexedShards.map { kmeans.centroids[$0.index] }

        self.shards = builtShards
        self.centroids = builtCentroids
        self.isBuilt = true
    }

    public func search(
        query: [Float],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        guard isBuilt, !shards.isEmpty, !centroids.isEmpty else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }

        guard let firstCentroid = centroids.first else {
            throw ANNSError.indexEmpty
        }
        guard query.count == firstCentroid.count else {
            throw ANNSError.dimensionMismatch(expected: firstCentroid.count, got: query.count)
        }

        let searchMetric = metric ?? configuration.metric

        let probeCount = min(nprobe, shards.count)
        let probeIndices = selectClosestCentroids(
            query: query,
            metric: searchMetric,
            count: probeCount
        )

        var mergedResults = BoundedSortedList<SearchResult>(capacity: k) {
            $0.score < $1.score
        }

        try await withThrowingTaskGroup(of: [SearchResult].self) { group in
            for shardIndex in probeIndices {
                let shard = shards[shardIndex]
                group.addTask {
                    try await shard.search(
                        query: query,
                        k: k,
                        filter: filter,
                        metric: metric
                    )
                }
            }

            for try await shardResults in group {
                mergedResults.insert(contentsOf: shardResults)
            }
        }

        return mergedResults.sortedElements
    }

    public func batchSearch(
        queries: [[Float]],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [[SearchResult]] {
        guard isBuilt, !shards.isEmpty, !centroids.isEmpty else {
            throw ANNSError.indexEmpty
        }
        guard !queries.isEmpty else {
            return []
        }

        return try await BatchExecution.run(
            over: queries,
            maxConcurrency: ProcessInfo.processInfo.activeProcessorCount
        ) { [self] query in
            try await searchForBatch(query: query, k: k, filter: filter, metric: metric)
        }
    }

    private func searchForBatch(
        query: [Float],
        k: Int,
        filter: _LegacySearchFilter?,
        metric: Metric?
    ) async throws -> [SearchResult] {
        guard k > 0 else {
            return []
        }
        guard let firstCentroid = centroids.first else {
            throw ANNSError.indexEmpty
        }
        guard query.count == firstCentroid.count else {
            throw ANNSError.dimensionMismatch(expected: firstCentroid.count, got: query.count)
        }

        let searchMetric = metric ?? configuration.metric

        let probeCount = min(shards.count, nprobe + 1)
        let probeIndices = selectClosestCentroids(
            query: query,
            metric: searchMetric,
            count: probeCount
        )
        // Batch mode uses a deeper per-shard candidate set. Keeping this above
        // the GPU beam-search cutoff steers toward the CPU/HNSW path, which is
        // currently more stable for batched sharded recall.
        let perShardK = max(k, max(257, min(512, max(configuration.efSearch * 4, k * 16))))

        var mergedResults = BoundedSortedList<SearchResult>(capacity: k) {
            $0.score < $1.score
        }

        try await withThrowingTaskGroup(of: [SearchResult].self) { group in
            for shardIndex in probeIndices {
                let shard = shards[shardIndex]
                group.addTask {
                    try await shard.search(
                        query: query,
                        k: perShardK,
                        filter: filter,
                        metric: metric
                    )
                }
            }

            for try await shardResults in group {
                mergedResults.insert(contentsOf: shardResults)
            }
        }

        return mergedResults.sortedElements
    }

    public var count: Int {
        get async {
            var total = 0
            for shard in shards {
                total += await shard.count
            }
            return total
        }
    }

    func shardSizes() async -> [Int] {
        var sizes: [Int] = []
        sizes.reserveCapacity(shards.count)

        for shard in shards {
            sizes.append(await shard.count)
        }

        return sizes
    }

    private func selectClosestCentroids(
        query: [Float],
        metric: Metric,
        count: Int
    ) -> [Int] {
        guard count > 0 else {
            return []
        }

        var best = BoundedSortedList<(index: Int, distance: Float)>(capacity: count) {
            $0.distance < $1.distance
        }

        for (index, centroid) in centroids.enumerated() {
            let distance = SIMDDistance.distance(query, centroid, metric: metric)
            best.insert((index, distance))
        }

        return best.sortedElements.map(\.index)
    }
}
