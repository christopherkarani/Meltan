import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Advanced.ShardedIndex Parallel Build Tests")
struct ShardedIndexParallelBuildTests {
    @Test("parallelBuildMatchesSequentialResults")
    func parallelBuildMatchesSequentialResults() async throws {
        let dim = 32
        let vectors = makeClusteredVectors(count: 800, dim: dim, clusters: 8)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let config = IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)

        let index = Advanced.ShardedIndex(numShards: 4, nprobe: 3, configuration: config)
        try await index.build(vectors: vectors, ids: ids)

        let sequential = try await buildSequentialReference(
            vectors: vectors,
            ids: ids,
            numShards: 4,
            nprobe: 3,
            configuration: config
        )

        var parallelHits = 0
        var sequentialHits = 0

        for i in 0..<20 {
            let query = vectors[i]
            let parallelResults = try await index.search(query: query, k: 10)
            let sequentialResults = try await searchSequentialReference(
                state: sequential,
                query: query,
                k: 10,
                metric: config.metric
            )

            #expect(parallelResults.count == sequentialResults.count)

            if parallelResults.contains(where: { $0.id == "v\(i)" }) {
                parallelHits += 1
            }
            if sequentialResults.contains(where: { $0.id == "v\(i)" }) {
                sequentialHits += 1
            }
        }

        let parallelRecall = Float(parallelHits) / 20.0
        let sequentialRecall = Float(sequentialHits) / 20.0
        #expect(abs(parallelRecall - sequentialRecall) <= 0.25)
    }

    @Test("parallelBuildCompletesWithoutError")
    func parallelBuildCompletesWithoutError() async throws {
        let vectors = makeClusteredVectors(count: 1200, dim: 32, clusters: 12)
        let ids = (0..<vectors.count).map { "v\($0)" }

        let index = Advanced.ShardedIndex(
            numShards: 8,
            nprobe: 4,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )

        try await index.build(vectors: vectors, ids: ids)
        #expect(await index.count == vectors.count)
    }

    @Test("parallelBuildTimingLogged")
    func parallelBuildTimingLogged() async throws {
        let vectors = makeClusteredVectors(count: 800, dim: 32, clusters: 8)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let config = IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)

        let parallel = Advanced.ShardedIndex(numShards: 4, nprobe: 3, configuration: config)

        let parallelStart = ContinuousClock.now
        try await parallel.build(vectors: vectors, ids: ids)
        let parallelDuration = parallelStart.duration(to: .now)

        let sequentialStart = ContinuousClock.now
        _ = try await buildSequentialReference(
            vectors: vectors,
            ids: ids,
            numShards: 4,
            nprobe: 3,
            configuration: config
        )
        let sequentialDuration = sequentialStart.duration(to: .now)

        let parallelSeconds = durationSeconds(parallelDuration)
        let sequentialSeconds = durationSeconds(sequentialDuration)
        let speedup = sequentialSeconds / max(parallelSeconds, 1e-9)
        print("Sharded build timing (parallel=\(parallelSeconds)s, sequential=\(sequentialSeconds)s, speedup=\(speedup)x)")

        #expect(parallelSeconds > 0)
        #expect(sequentialSeconds > 0)
    }

    private struct SequentialState {
        let centroids: [[Float]]
        let shards: [GraphIndex]
        let nprobe: Int
    }

    private func buildSequentialReference(
        vectors: [[Float]],
        ids: [String],
        numShards: Int,
        nprobe: Int,
        configuration: IndexConfiguration
    ) async throws -> SequentialState {
        let effectiveShards = min(max(1, numShards), vectors.count)
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

        var builtShards: [GraphIndex] = []
        var builtCentroids: [[Float]] = []

        for shardIndex in 0..<effectiveShards {
            guard !shardVectors[shardIndex].isEmpty else {
                continue
            }

            var shardConfiguration = configuration
            shardConfiguration.degree = min(
                configuration.degree,
                max(1, shardVectors[shardIndex].count - 1)
            )

            let shard = GraphIndex(configuration: shardConfiguration)
            try await shard.build(vectors: shardVectors[shardIndex], ids: shardIDs[shardIndex])
            builtShards.append(shard)
            builtCentroids.append(kmeans.centroids[shardIndex])
        }

        return SequentialState(centroids: builtCentroids, shards: builtShards, nprobe: nprobe)
    }

    private func searchSequentialReference(
        state: SequentialState,
        query: [Float],
        k: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        let centroidDistances = state.centroids.enumerated().map { (index, centroid) in
            (index, SIMDDistance.distance(query, centroid, metric: metric))
        }.sorted { $0.1 < $1.1 }

        let probeCount = min(state.nprobe, state.shards.count)
        let probeIndices = centroidDistances.prefix(probeCount).map { $0.0 }

        var mergedResults: [SearchResult] = []
        mergedResults.reserveCapacity(probeCount * k)

        for shardIndex in probeIndices {
            let shardResults = try await state.shards[shardIndex].search(query: query, k: k, metric: metric)
            mergedResults.append(contentsOf: shardResults)
        }

        mergedResults.sort { $0.score < $1.score }
        return Array(mergedResults.prefix(k))
    }

    private func makeClusteredVectors(count: Int, dim: Int, clusters: Int) -> [[Float]] {
        (0..<count).map { i in
            let cluster = i % clusters
            return (0..<dim).map { d in
                let center: Float = d == (cluster % dim) ? 1.0 : 0.0
                let noiseSeed = Float((i * dim) + d)
                return center + sin(noiseSeed * 0.031) * 0.01
            }
        }
    }

    private func durationSeconds(_ duration: Duration) -> Double {
        let c = duration.components
        return Double(c.seconds) + (Double(c.attoseconds) / 1_000_000_000_000_000_000)
    }
}
