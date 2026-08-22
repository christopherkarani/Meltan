import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Range Search Tests")
struct RangeSearchTests {
    @Test("Range search returns only results within threshold")
    func rangeSearchReturnsWithinThreshold() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 100)
        let ids = (0..<100).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.rangeSearch(query: vectors[7], maxDistance: 0.5)

        #expect(!results.isEmpty)
        #expect(results.allSatisfy { $0.score <= 0.5 })
    }

    @Test("Range search finds exact match with tight threshold")
    func rangeSearchFindsExactMatch() async throws {
        let dim = 16
        let vectors = makeVectors(count: 50, dim: dim, seedOffset: 200)
        let ids = (0..<50).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        let target = makeVectors(count: 1, dim: dim, seedOffset: 999)[0]
        try await index.insert(target, id: "target")

        let results = try await index.rangeSearch(query: target, maxDistance: 0.01)
        #expect(results.contains { $0.id == "target" })
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.081) + cos(i * 0.057)
            }
        }
    }
}
