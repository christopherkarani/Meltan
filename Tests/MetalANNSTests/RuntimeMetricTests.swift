import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Runtime Metric Selection Tests")
struct RuntimeMetricTests {
    @Test("Search with different metric returns valid results")
    func searchWithDifferentMetric() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 300)
        let ids = (0..<100).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.search(query: vectors[12], k: 10, metric: .l2)

        #expect(!results.isEmpty)
        #expect(results.allSatisfy { $0.score >= 0 })

        for i in 1..<results.count {
            #expect(results[i - 1].score <= results[i].score)
        }
    }

    @Test("Default metric matches build metric")
    func defaultMetricMatchesBuildMetric() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 301)
        let ids = (0..<100).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        let implicit = try await index.search(query: vectors[20], k: 10)
        let explicit = try await index.search(query: vectors[20], k: 10, metric: .cosine)

        #expect(implicit.map(\.id) == explicit.map(\.id))
        #expect(implicit.map(\.internalID) == explicit.map(\.internalID))
        #expect(implicit.map(\.score) == explicit.map(\.score))
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.069) + cos(i * 0.041)
            }
        }
    }
}
