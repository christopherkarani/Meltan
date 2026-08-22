import Foundation
import MetalANNSFixtures
import Testing

@testable import MetalANNS

@Suite("Concurrent Search Tests")
struct ConcurrentSearchTests {
    @Test("batchSearchMatchesSequential")
    func batchSearchMatchesSequential() async throws {
        let vectors = Fixtures.syntheticVectors(count: 100, dim: 16, seedOffset: 0)
        let ids = (0..<100).map { "v_\($0)" }
        let queries = Fixtures.syntheticVectors(count: 10, dim: 16, seedOffset: 10_000)

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        var sequential: [[SearchResult]] = []
        sequential.reserveCapacity(queries.count)
        for query in queries {
            sequential.append(try await index.search(query: query, k: 5))
        }

        let concurrent = try await index.batchSearch(queries: queries, k: 5)
        #expect(concurrent.count == sequential.count)

        for i in 0..<queries.count {
            let sequentialIDs = Set(sequential[i].map(\.id))
            let concurrentIDs = Set(concurrent[i].map(\.id))
            #expect(sequentialIDs == concurrentIDs)
        }
    }

    @Test("batchSearchHandlesLargeQueryCount")
    func batchSearchHandlesLargeQueryCount() async throws {
        let vectors = Fixtures.syntheticVectors(count: 200, dim: 32, seedOffset: 0)
        let ids = (0..<200).map { "v_\($0)" }
        let queries = Fixtures.syntheticVectors(count: 50, dim: 32, seedOffset: 20_000)

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.batchSearch(queries: queries, k: 10)
        #expect(results.count == 50)
        #expect(results.allSatisfy { $0.count == 10 })
        #expect(results.allSatisfy { !$0.isEmpty })
    }
}
