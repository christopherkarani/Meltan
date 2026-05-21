import Foundation
import Testing
@testable import MetalANNS

@Suite("Batch Insert Tests")
struct BatchInsertTests {
    @Test("Batch insert produces searchable results")
    func batchInsertRecall() async throws {
        let dim = 32
        let baseVectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let baseIDs = (0..<100).map { "base_\($0)" }

        let insertedVectors = makeVectors(count: 50, dim: dim, seedOffset: 10_000)
        let insertedIDs = (0..<50).map { "new_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)
        try await index.batchInsert(insertedVectors, ids: insertedIDs)

        let recall = try await selfRecall(
            index: index,
            vectors: insertedVectors,
            ids: insertedIDs,
            k: 10
        )

        #expect(recall >= 0.70)
        #expect(await index.count == 150)
    }

    @Test("Batch insert matches sequential insert quality")
    func batchInsertMatchesSequential() async throws {
        let dim = 32
        let baseVectors = makeVectors(count: 100, dim: dim, seedOffset: 100)
        let baseIDs = (0..<100).map { "base_\($0)" }
        let insertedVectors = makeVectors(count: 50, dim: dim, seedOffset: 20_000)
        let insertedIDs = (0..<50).map { "new_\($0)" }

        let batchIndex = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await batchIndex.build(vectors: baseVectors, ids: baseIDs)
        try await batchIndex.batchInsert(insertedVectors, ids: insertedIDs)

        let sequentialIndex = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await sequentialIndex.build(vectors: baseVectors, ids: baseIDs)
        for (offset, vector) in insertedVectors.enumerated() {
            try await sequentialIndex.insert(vector, id: insertedIDs[offset])
        }

        let batchRecall = try await selfRecall(index: batchIndex, vectors: insertedVectors, ids: insertedIDs, k: 10)
        let sequentialRecall = try await selfRecall(
            index: sequentialIndex,
            vectors: insertedVectors,
            ids: insertedIDs,
            k: 10
        )

        #expect(batchRecall + 0.10 >= sequentialRecall)
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
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }
}
