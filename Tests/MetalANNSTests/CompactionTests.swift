import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Compaction Tests")
struct CompactionTests {
    @Test("Compact reduces node count after deletions")
    func compactReducesMemory() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "vec_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        for id in ids.prefix(50) {
            try await index.delete(id: id)
        }

        #expect(await index.count == 50)

        try await index.compact()

        #expect(await index.count == 50)

        let results = try await index.search(query: vectors[70], k: 10)
        #expect(results.contains { $0.id == "vec_70" })

        do {
            try await index.delete(id: "vec_0")
            #expect(Bool(false), "Expected idNotFound after compaction")
        } catch let error as ANNSError {
            if case .idNotFound = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.idNotFound but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.idNotFound")
        }
    }

    @Test("Compact maintains search recall")
    func compactMaintainsRecall() async throws {
        let dim = 32
        let vectors = makeVectors(count: 200, dim: dim, seedOffset: 50)
        let ids = (0..<200).map { "vec_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let queryVectors = Array(vectors.prefix(10))
        let queryIDs = Array(ids.prefix(10))
        let baselineRecall = try await selfRecall(index: index, vectors: queryVectors, ids: queryIDs, k: 10)

        for id in ids[100..<200] {
            try await index.delete(id: id)
        }

        try await index.compact()

        let postRecall = try await selfRecall(index: index, vectors: queryVectors, ids: queryIDs, k: 10)
        #expect(postRecall >= baselineRecall * 0.80)
    }

    @Test("Compact preserves UInt64-keyed entries in mixed ID indexes")
    func compactPreservesUInt64Entries() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 50, dim: dim, seedOffset: 2_000)
        let baseIDs = (0..<50).map { "base-\($0)" }
        let numericVector = makeVectors(count: 1, dim: dim, seedOffset: 99_999)[0]

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)
        try await index.insert(numericVector, numericID: 42)
        try await index.delete(id: "base-0")

        try await index.compact()

        let results = try await index.search(query: numericVector, k: 5)
        #expect(results.contains { $0.numericID == 42 })
    }

    private func selfRecall(
        index: GraphIndex,
        vectors: [[Float]],
        ids: [String],
        k: Int
    ) async throws -> Float {
        var hits = 0
        for (offset, query) in vectors.enumerated() {
            let results = try await index.search(query: query, k: k)
            if results.contains(where: { $0.id == ids[offset] }) {
                hits += 1
            }
        }
        return Float(hits) / Float(vectors.count)
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.137) + cos(i * 0.089)
            }
        }
    }
}
