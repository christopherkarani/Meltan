import Foundation
import Testing

@testable import MetalANNS

@Suite("Advanced.StreamingIndex Persistence Tests")
struct StreamingIndexPersistenceTests {
    @Test("Save and load roundtrip count")
    func saveAndLoadEmpty() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 20,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<5).map { makeVector(row: $0, dim: 16) }
        let ids = (0..<5).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("streaming.db").path))
        let loaded = try await Advanced.StreamingIndex.load(from: dir)

        #expect(await loaded.count == 5)
    }

    @Test("Search after load")
    func searchAfterLoad() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 20,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<20).map { makeVector(row: 100 + $0, dim: 16) }
        let ids = (0..<20).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let loaded = try await Advanced.StreamingIndex.load(from: dir)

        let results = try await loaded.search(query: vectors[7], k: 3)
        #expect(results.contains(where: { $0.id == "v7" }))
    }

    @Test("Save auto-flushes pending delta")
    func saveRequiresFlush() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 100,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<10).map { makeVector(row: 1_000 + $0, dim: 16) }
        let ids = (0..<10).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let loaded = try await Advanced.StreamingIndex.load(from: dir)
        #expect(await loaded.count == 10)
    }

    @Test("Save creates streaming.db and removes JSON sidecar")
    func saveCreatesStreamingDB() async throws {
        let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
        let index = Advanced.StreamingIndex(config: config)

        for i in 0..<20 {
            let vector = makeVector(row: i, dim: 8)
            try await index.insert(vector, id: "vec-\(i)")
        }
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)

        let dbPath = dir.appendingPathComponent("streaming.db").path
        #expect(
            FileManager.default.fileExists(atPath: dbPath),
            "Expected streaming.db to be created"
        )

        let jsonPath = dir.appendingPathComponent("streaming.meta.json").path
        #expect(
            !FileManager.default.fileExists(atPath: jsonPath),
            "Should no longer create streaming.meta.json"
        )
    }

    @Test("Save/load roundtrip via SQLite")
    func saveLoadRoundtripViaSQLite() async throws {
        let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
        let index = Advanced.StreamingIndex(config: config)

        for i in 0..<30 {
            let vector = makeVector(row: i, dim: 8)
            try await index.insert(vector, id: "vec-\(i)")
        }
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let loaded = try await Advanced.StreamingIndex.load(from: dir)
        let count = await loaded.count
        #expect(count == 30)
    }

    @Test("mergedVectorsAreEvictedFromHistory")
    func mergedVectorsAreEvictedFromHistory() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 10,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<15).map { makeVector(row: 40_000 + $0, dim: 8) }
        let ids = (0..<15).map { "m-\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let persistedIDs = try persistedHistoryIDs(at: dir)
        #expect(
            persistedIDs.isEmpty,
            "Persisted history should only contain post-merge pending IDs, not merged IDs"
        )
    }

    @Test("deletedVectorsAreRemovedFromHistory")
    func deletedVectorsAreRemovedFromHistory() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 10,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<5).map { makeVector(row: 50_000 + $0, dim: 8) }
        let ids = (0..<5).map { "d-\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.delete(id: "d-1")
        try await index.delete(id: "d-3")

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let persistedIDs = try persistedHistoryIDs(at: dir)
        #expect(!persistedIDs.contains("d-1"))
        #expect(!persistedIDs.contains("d-3"))
    }

    private func tempDirectoryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-streaming-\(UUID().uuidString)")
    }

    private func persistedHistoryIDs(at dir: URL) throws -> [String] {
        let dbPath = dir.appendingPathComponent("streaming.db").path
        let db = try StreamingDatabase(path: dbPath)
        let (_, ids) = try db.loadAllVectors()
        return ids
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.149) + cos(i * 0.063)
        }
    }
}
