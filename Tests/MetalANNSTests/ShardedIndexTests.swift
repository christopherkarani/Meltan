import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Sharded Index Tests")
struct ShardedIndexTests {
    @Test("Sharded search achieves reasonable recall")
    func shardedSearchRecall() async throws {
        let dim = 32
        let vectors = makeClusteredVectors(count: 500, dim: dim, clusters: 8)
        let ids = (0..<500).map { "v\($0)" }

        let index = Advanced.ShardedIndex(
            numShards: 8,
            nprobe: 3,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )
        try await index.build(vectors: vectors, ids: ids)

        var hits = 0
        for i in 0..<20 {
            let results = try await index.search(query: vectors[i], k: 10)
            if results.contains(where: { $0.id == "v\(i)" }) {
                hits += 1
            }
        }

        let recall = Float(hits) / 20.0
        #expect(recall >= 0.70)
    }

    @Test("Sharded index distributes vectors across shards")
    func shardedDistribution() async throws {
        let dim = 16
        let vectors = makeClusteredVectors(count: 200, dim: dim, clusters: 4)
        let ids = (0..<200).map { "v\($0)" }

        let index = Advanced.ShardedIndex(
            numShards: 4,
            nprobe: 2,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )
        try await index.build(vectors: vectors, ids: ids)

        #expect(await index.count == 200)

        let sizes = await index.shardSizes()
        #expect(sizes.count == 4)
        #expect(sizes.allSatisfy { $0 > 0 })
        #expect(sizes.reduce(0, +) == 200)
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
}
