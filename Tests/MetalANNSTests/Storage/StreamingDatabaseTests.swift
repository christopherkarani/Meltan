import Foundation
import Testing

@testable import MetalANNS

@Suite("StreamingDatabase Tests")
struct StreamingDatabaseTests {
    private func tempDBPath() -> String {
        NSTemporaryDirectory() + "streaming-test-\(UUID().uuidString).db"
    }

    @Test("Insert and fetch vectors")
    func insertAndFetchVectors() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        let vectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        let ids = ["a", "b", "c"]

        try db.insertVectors(vectors, ids: ids)

        let (loadedVectors, loadedIDs) = try db.loadAllVectors()
        #expect(loadedIDs == ids)
        #expect(loadedVectors.count == 3)
        #expect(loadedVectors[0] == [1.0, 2.0, 3.0])
    }

    @Test("Incremental insert")
    func incrementalInsert() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        try db.insertVectors([[1.0, 2.0]], ids: ["first"])
        try db.insertVectors([[3.0, 4.0]], ids: ["second"])

        let (vectors, ids) = try db.loadAllVectors()
        #expect(ids.count == 2)
        #expect(vectors.count == 2)
    }

    @Test("Mark deleted IDs")
    func markDeleted() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        try db.insertVectors([[1.0], [2.0], [3.0]], ids: ["a", "b", "c"])
        try db.markDeleted(ids: ["b"])

        let deletedIDs = try db.loadDeletedIDs()
        #expect(deletedIDs == Set(["b"]))
    }

    @Test("Save and load config")
    func saveAndLoadConfig() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        let config = StreamingConfiguration(deltaCapacity: 500, mergeStrategy: .blocking)
        try db.saveConfig(config)

        let loaded = try db.loadConfig()
        #expect(loaded?.deltaCapacity == 500)
    }

    @Test("Save and load per-vector metadata")
    func saveAndLoadPerVectorMetadata() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        try db.insertVectors([[1.0]], ids: ["vec-1"])
        try db.saveVectorMetadata(id: "vec-1", metadata: ["color": "red", "score": "0.95"])

        let loaded = try db.loadVectorMetadata(id: "vec-1")
        #expect(loaded?["color"] == "red")
        #expect(loaded?["score"] == "0.95")
    }
}
