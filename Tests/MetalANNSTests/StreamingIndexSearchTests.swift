import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Advanced.StreamingIndex Search Tests")
struct StreamingIndexSearchTests {
    @Test("Search finds base and delta")
    func searchFindsBaseAndDelta() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 200,
                mergeStrategy: .blocking
            ))

        let baseVectors = (0..<100).map { makeVector(row: $0, dim: 16) }
        let baseIDs = (0..<100).map { "b\($0)" }
        try await index.batchInsert(baseVectors, ids: baseIDs)
        try await index.flush()

        let deltaVectors = (0..<50).map { makeVector(row: 10_000 + $0, dim: 16) }
        let deltaIDs = (0..<50).map { "d\($0)" }
        try await index.batchInsert(deltaVectors, ids: deltaIDs)

        let results = try await index.search(query: deltaVectors[7], k: 5)
        #expect(results.contains(where: { $0.id == "d7" }))
    }

    @Test("Recall after merge")
    func recallAfterMerge() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 100,
                mergeStrategy: .blocking
            ))

        var allVectors: [[Float]] = []
        var allIDs: [String] = []
        for batch in 0..<3 {
            let vectors = (0..<100).map { makeVector(row: batch * 100 + $0, dim: 16) }
            let ids = (0..<100).map { "v\(batch)-\($0)" }
            allVectors.append(contentsOf: vectors)
            allIDs.append(contentsOf: ids)
            try await index.batchInsert(vectors, ids: ids)
        }

        try await index.flush()

        var hits = 0
        let queryCount = 20
        for i in 0..<queryCount {
            let query = allVectors[i * 7]
            let expectedID = allIDs[i * 7]
            let results = try await index.search(query: query, k: 10)
            if results.contains(where: { $0.id == expectedID }) {
                hits += 1
            }
        }

        let recall = Float(hits) / Float(queryCount)
        #expect(recall > 0.90)
    }

    @Test("Search with filter forwards")
    func searchWithFilterForwards() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 100,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<50).map { makeVector(row: $0, dim: 16) }
        let ids = (0..<50).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        for i in 0..<10 {
            try await index.setMetadata("tag", value: "hot", for: "v\(i)")
        }

        let results = try await index.search(
            query: vectors[3],
            k: 20,
            filter: .equals(column: "tag", value: "hot")
        )

        #expect(!results.isEmpty)
        for result in results {
            let indexValue = Int(result.id.dropFirst()) ?? -1
            #expect((0..<10).contains(indexValue))
        }
    }

    @Test("Range search covers all")
    func rangeSearchCoversAll() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 100,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<100).map { makeVector(row: 20_000 + $0, dim: 16) }
        let ids = (0..<100).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.flush()

        let results = try await index.rangeSearch(query: vectors[0], maxDistance: 10.0, limit: 500)
        #expect(!results.isEmpty)
    }

    @Test("rangeSearch with maxDistance 0 returns exact matches")
    func rangeSearchZeroDistanceReturnsExactMatch() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 50,
                mergeStrategy: .blocking,
                indexConfiguration: IndexConfiguration(degree: 4, metric: .l2)
            ))

        let dim = 8
        let vectors = (0..<10).map { makeVector(row: 30_000 + $0, dim: dim) }
        let ids = (0..<10).map { "v-\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        let results = try await index.rangeSearch(query: vectors[3], maxDistance: 0.0, limit: 10)
        #expect(
            results.contains(where: { $0.id == "v-3" }),
            "rangeSearch(maxDistance: 0) must return the exact-match vector"
        )
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.173) + cos(i * 0.071)
        }
    }
}
