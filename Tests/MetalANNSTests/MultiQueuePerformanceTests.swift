import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Multi-Queue Performance Tests")
struct MultiQueuePerformanceTests {
    @Test("shardedBuildSpeedup")
    func shardedBuildSpeedup() async throws {
        let vectors = makeClusteredVectors(count: 2400, dim: 32, clusters: 12)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let config = IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)

        let parallel = Advanced.ShardedIndex(numShards: 8, nprobe: 4, configuration: config)

        let parallelStart = ContinuousClock.now
        try await parallel.build(vectors: vectors, ids: ids)
        let parallelSeconds = durationSeconds(parallelStart.duration(to: .now))

        let sequentialStart = ContinuousClock.now
        _ = try await buildSequentialReference(
            vectors: vectors,
            ids: ids,
            numShards: 8,
            nprobe: 4,
            configuration: config
        )
        let sequentialSeconds = durationSeconds(sequentialStart.duration(to: .now))

        let speedup = sequentialSeconds / max(parallelSeconds, 1e-9)
        print(
            "Sharded build speedup: parallel=\(parallelSeconds)s sequential=\(sequentialSeconds)s speedup=\(speedup)x")
        #expect(parallelSeconds > 0)
        #expect(sequentialSeconds > 0)
        #expect(speedup > 0)
    }

    @Test("batchSearchQPS")
    func batchSearchQPS() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            let context = try Require.metalContext()
            let vectors = makeVectors(count: 10_000, dim: 32, seedOffset: 0)
            let ids = (0..<vectors.count).map { "v\($0)" }
            let queries = Array(vectors.prefix(200))
            let index = GraphIndex(
                configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96),
                context: context
            )
            try await index.build(vectors: vectors, ids: ids)

            let start = ContinuousClock.now
            _ = try await index.batchSearch(queries: queries, k: 10)
            let elapsed = durationSeconds(start.duration(to: .now))
            let qps = Double(queries.count) / max(elapsed, 1e-9)
            print("Batch search QPS: \(qps)")
            #expect(qps > 0)
        #endif
    }

    @Test("shardedSearchQPS")
    func shardedSearchQPS() async throws {
        let vectors = makeClusteredVectors(count: 2000, dim: 32, clusters: 10)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let queries = Array(vectors.prefix(100))
        let index = Advanced.ShardedIndex(
            numShards: 4,
            nprobe: 4,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )
        try await index.build(vectors: vectors, ids: ids)

        let start = ContinuousClock.now
        for query in queries {
            _ = try await index.search(query: query, k: 10)
        }
        let elapsed = durationSeconds(start.duration(to: .now))
        let qps = Double(queries.count) / max(elapsed, 1e-9)
        print("Sharded search QPS: \(qps)")
        #expect(qps > 0)
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

    private func durationSeconds(_ duration: Duration) -> Double {
        let c = duration.components
        return Double(c.seconds) + (Double(c.attoseconds) / 1_000_000_000_000_000_000)
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

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }
}
