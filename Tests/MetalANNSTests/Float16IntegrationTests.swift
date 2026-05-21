import Foundation
import MetalANNS
import MetalANNSCore
import Testing

@Suite("Float16 Integration")
struct Float16IntegrationTests {
    @Test func float16FullLifecycle() async throws {
        let config = IndexConfiguration(degree: 8, metric: .cosine, useFloat16: true)
        let index = GraphIndex(configuration: config)

        let dim = 32
        let n = 100
        let vectors = (0..<n).map { i in
            seededVector(dim: dim, seed: 100 + UInt64(i))
        }
        let ids = (0..<n).map { "v_\($0)" }

        try await index.build(vectors: vectors, ids: ids)

        let query = seededVector(dim: dim, seed: 999)
        let results = try await index.search(query: query, k: 5)
        #expect(results.count == 5)
        #expect(results.allSatisfy { !$0.id.isEmpty })

        let newVector = seededVector(dim: dim, seed: 8888)
        try await index.insert(newVector, id: "v_new")

        let countAfterInsert = await index.count
        #expect(countAfterInsert == n + 1)

        let results2 = try await index.search(query: newVector, k: 5)
        #expect(results2.count == 5)
        #expect(results2.contains { $0.id == "v_new" })
    }

    @Test func float16SaveLoadPreservesData() async throws {
        let config = IndexConfiguration(degree: 8, metric: .l2, useFloat16: true)
        let index = GraphIndex(configuration: config)

        let dim = 16
        let n = 50
        let vectors = (0..<n).map { i in
            seededVector(dim: dim, seed: 200 + UInt64(i))
        }
        let ids = (0..<n).map { "v_\($0)" }

        try await index.build(vectors: vectors, ids: ids)

        let query = vectors[0]
        let beforeResults = try await index.search(query: query, k: 3)
        #expect(beforeResults.count == 3)

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("float16_test_\(UUID().uuidString)")
        let fileURL = tempDir.appendingPathComponent("index.mann")
        try await index.save(to: fileURL)

        let loaded = try await GraphIndex.load(from: fileURL)

        let afterResults = try await loaded.search(query: query, k: 3)
        #expect(afterResults.count == 3)
        #expect(beforeResults.map(\.id) == afterResults.map(\.id))

        try? FileManager.default.removeItem(at: tempDir)
    }
}
