import Foundation
import Testing
@testable import MetalANNS
import MetalANNSCore

@Suite("GraphIndex Public API Tests")
struct ANNSIndexTests {
    @Test("Build and search returns mapped external IDs")
    func buildAndSearch() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "vec_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.search(query: vectors[0], k: 5)

        #expect(results.count == 5)
        #expect(results[0].id == "vec_0")
        #expect(abs(results[0].score) < 1e-4)
        #expect(results.allSatisfy { !$0.id.isEmpty })
    }

    @Test("Insert then search finds inserted vectors")
    func insertAndSearch() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 50, dim: dim, seedOffset: 0)
        let baseIDs = (0..<50).map { "vec_\($0)" }
        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)

        let insertedVectors = makeVectors(count: 5, dim: dim, seedOffset: 10_000)
        for i in 0..<5 {
            try await index.insert(insertedVectors[i], id: "new_\(i)")
        }

        for i in 0..<5 {
            let results = try await index.search(query: insertedVectors[i], k: 1)
            #expect(results.count == 1)
            #expect(results[0].id == "new_\(i)")
        }

        #expect(await index.count == 55)
    }

    @Test("Insert and search with UInt64 ID returns numericID in results")
    func insertAndSearchWithUInt64IDs() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 20, dim: dim, seedOffset: 1_000)
        let baseIDs = (0..<20).map { "base-\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)

        // Insert with UInt64 key.
        let newVector = makeVectors(count: 1, dim: dim, seedOffset: 9_999)[0]
        try await index.insert(newVector, numericID: 9_001)

        // Search using the inserted vector as query — first result should be itself.
        let results = try await index.search(query: newVector, k: 1)
        #expect(results.count == 1)
        #expect(
            results[0].numericID == 9_001,
            "Expected numericID 9001, got \(String(describing: results[0].numericID))"
        )
        // String ID for a UInt64-keyed result should be empty.
        #expect(results[0].id.isEmpty, "UInt64-inserted nodes have no String ID")
    }

    @Test("String-keyed results have nil numericID; UInt64-keyed results have nil String id")
    func stringAndUInt64ResultsCoexist() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 10, dim: dim, seedOffset: 500)
        let baseIDs = (0..<10).map { "str-\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)
        try await index.insert(makeVectors(count: 1, dim: dim, seedOffset: 88_888)[0], numericID: 42)

        // Search with broad k to get both String and UInt64 results.
        let results = try await index.search(
            query: makeVectors(count: 1, dim: dim, seedOffset: 88_888)[0],
            k: 5
        )

        let numericResults = results.filter { $0.numericID != nil }
        let stringResults = results.filter { !$0.id.isEmpty }

        #expect(!numericResults.isEmpty, "At least one UInt64-keyed result expected")
        // String-keyed results must not have a numericID.
        for result in stringResults {
            #expect(result.numericID == nil, "String-keyed result '\(result.id)' should have nil numericID")
        }
    }

    @Test("Save/load roundtrip preserves UInt64-keyed entries")
    func saveLoadRoundtripPreservesUInt64IDs() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 20, dim: dim, seedOffset: 3_500)
        let baseIDs = (0..<20).map { "node-\($0)" }
        let numericVector = makeVectors(count: 1, dim: dim, seedOffset: 123_456)[0]

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)
        try await index.insert(numericVector, numericID: UInt64.max - 1)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-numeric-roundtrip-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let tempMetaURL = URL(fileURLWithPath: tempURL.path + ".meta.json")
        let tempDBURL = URL(fileURLWithPath:
            tempURL.deletingPathExtension().appendingPathExtension("db").path)
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
            try? FileManager.default.removeItem(at: tempDBURL)
        }

        try await index.save(to: tempURL)
        let loaded = try await GraphIndex.load(from: tempURL)
        let results = try await loaded.search(query: numericVector, k: 3)
        #expect(results.contains { $0.numericID == UInt64.max - 1 })
    }

    @Test("Delete excludes soft-deleted IDs from search")
    func deleteAndSearch() async throws {
        let dim = 16
        let vectors = makeVectors(count: 50, dim: dim, seedOffset: 0)
        let ids = (0..<50).map { "vec_\($0)" }
        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        try await index.delete(id: "vec_0")
        try await index.delete(id: "vec_5")

        let results = try await index.search(query: vectors[1], k: 50)
        let resultIDs = Set(results.map(\.id))
        #expect(!resultIDs.contains("vec_0"))
        #expect(!resultIDs.contains("vec_5"))
        #expect(await index.count == 48)
    }

    @Test("Save and load round-trips lifecycle state")
    func saveAndLoadLifecycle() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let baseIDs = (0..<100).map { "vec_\($0)" }
        let insertedVectors = makeVectors(count: 5, dim: dim, seedOffset: 20_000)

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)
        for i in 0..<5 {
            try await index.insert(insertedVectors[i], id: "new_\(i)")
        }
        try await index.delete(id: "vec_0")
        try await index.delete(id: "vec_10")

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-public-api-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let tempMetaURL = URL(fileURLWithPath: tempURL.path + ".meta.json")
        let tempDBURL = URL(fileURLWithPath:
            tempURL.deletingPathExtension().appendingPathExtension("db").path)
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
            try? FileManager.default.removeItem(at: tempDBURL)
        }

        let before = try await index.search(query: baseVectors[12], k: 10)
        try await index.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempDBURL.path))

        let loaded = try await GraphIndex.load(from: tempURL)
        let after = try await loaded.search(query: baseVectors[12], k: 10)

        #expect(before.map(\.id) == after.map(\.id))
        #expect(before.map(\.internalID) == after.map(\.internalID))
        let originalCount = await index.count
        let loadedCount = await loaded.count
        #expect(originalCount == loadedCount)
    }

    @Test("Save creates SQLite sidecar")
    func saveCreatesDBFile() async throws {
        let index = GraphIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append(seededVector(dim: 8))
            ids.append("node-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
        try await index.save(to: url)

        #expect(FileManager.default.fileExists(atPath: url.path))

        let dbPath = URL(fileURLWithPath: url.deletingPathExtension().path)
            .appendingPathExtension("db").path
        #expect(FileManager.default.fileExists(atPath: dbPath))

        #expect(!FileManager.default.fileExists(atPath: url.path + ".meta.json"))
    }

    @Test("Load roundtrip prefers SQLite when fresh")
    func loadFromSQLiteRoundtrip() async throws {
        let index = GraphIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append(seededVector(dim: 8))
            ids.append("node-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
        try await index.save(to: url)

        let metaJSON = URL(fileURLWithPath: url.path + ".meta.json")
        try? FileManager.default.removeItem(at: metaJSON)

        let loaded = try await GraphIndex.load(from: url)
        #expect(await loaded.count == 50)

        let query = vectors[0]
        let results = try await loaded.search(query: query, k: 5)
        #expect(results.count == 5)
        #expect(results.allSatisfy { $0.id.hasPrefix("node-") })
    }

    @Test("Load falls back to JSON when DB missing")
    func loadFallsBackToJSONSidecar() async throws {
        let index = GraphIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append(seededVector(dim: 8))
            ids.append("legacy-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("legacy.anns")
        try await index.save(to: url)

        let dbPath = url.deletingPathExtension().appendingPathExtension("db").path
        try? FileManager.default.removeItem(atPath: dbPath)

        struct LegacyPersistedMetadata: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }
        let legacyMeta = LegacyPersistedMetadata(
            configuration: .default,
            softDeletion: SoftDeletion(),
            metadataStore: nil
        )
        let metaJSON = try JSONEncoder().encode(legacyMeta)
        let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
        try metaJSON.write(to: metaURL, options: Data.WritingOptions.atomic)

        let loaded = try await GraphIndex.load(from: url)
        #expect(await loaded.count == 50)
    }

    @Test("Backward compatibility: old JSON sidecar loads and re-saves as SQLite")
    func backwardCompatLoadThenSaveUpgrades() async throws {
        let vectors: [[Float]] = (0..<40).map { i in
            (0..<8).map { j in Float(i * 8 + j) * 0.01 }
        }
        let ids = (0..<40).map { "compat-\($0)" }
        let index = GraphIndex(configuration: .default)
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "compat-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
        let dbURL = URL(fileURLWithPath: url.deletingPathExtension().appendingPathExtension("db").path)

        try await index.save(to: url)
        #expect(FileManager.default.fileExists(atPath: dbURL.path), "Expected .db after save")

        try FileManager.default.removeItem(at: dbURL)

        struct LegacyMeta: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }

        let metaData = try JSONEncoder().encode(
            LegacyMeta(configuration: .default, softDeletion: SoftDeletion(), metadataStore: nil)
        )
        let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
        try metaData.write(to: metaURL, options: .atomic)

        let loaded = try await GraphIndex.load(from: url)
        #expect(await loaded.count == 40, "Load from JSON sidecar should restore 40 vectors")

        try await loaded.save(to: url)
        #expect(FileManager.default.fileExists(atPath: dbURL.path), "Re-save should create .db")

        let reloaded = try await GraphIndex.load(from: url)
        #expect(await reloaded.count == 40, "Load from SQLite should restore 40 vectors")

        let results = try await reloaded.search(query: vectors[1], k: 5)
        #expect(!results.isEmpty, "Search should return results")
        #expect(results.allSatisfy { $0.id.hasPrefix("compat-") }, "Search should return compat IDs")
    }

    @Test("Load falls back when DB is stale or unreadable")
    func loadFallsBackWhenDBInvalid() async throws {
        let index = GraphIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append(seededVector(dim: 8))
            ids.append("corrupt-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("corrupt.anns")
        try await index.save(to: url)

        let dbURL = URL(fileURLWithPath: url.deletingPathExtension().appendingPathExtension("db").path)
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: dbURL, options: .atomic)

        struct LegacyMeta: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }
        let fallbackJSON = try JSONEncoder().encode(
            LegacyMeta(
                configuration: .default,
                softDeletion: SoftDeletion(),
                metadataStore: nil
            )
        )
        try fallbackJSON.write(
            to: URL(fileURLWithPath: url.path + ".meta.json"),
            options: Data.WritingOptions.atomic
        )

        let loaded = try await GraphIndex.load(from: url)
        #expect(await loaded.count == 50)
    }

    @Test("Load falls back to JSON when legacy meta.db is unreadable")
    func loadFallsBackWhenLegacyMetaDBCorrupt() async throws {
        let index = GraphIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append(seededVector(dim: 8))
            ids.append("legacy-corrupt-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("legacy-corrupt.anns")
        try await index.save(to: url)

        let dbPath = url.deletingPathExtension().appendingPathExtension("db").path
        try? FileManager.default.removeItem(atPath: dbPath)

        struct LegacyMeta: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }
        let fallbackJSON = try JSONEncoder().encode(
            LegacyMeta(
                configuration: .default,
                softDeletion: SoftDeletion(),
                metadataStore: nil
            )
        )
        try fallbackJSON.write(
            to: URL(fileURLWithPath: url.path + ".meta.json"),
            options: .atomic
        )
        try Data([0x00, 0x01, 0x02, 0x03]).write(
            to: URL(fileURLWithPath: url.path + ".meta.db"),
            options: .atomic
        )

        let loaded = try await GraphIndex.load(from: url)
        #expect(await loaded.count == 50)
    }

    @Test("Save removes stale legacy metadata sidecars")
    func saveRemovesLegacyMetadataSidecars() async throws {
        let index = GraphIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append(seededVector(dim: 8))
            ids.append("cleanup-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("cleanup.anns")
        let metaJSONURL = URL(fileURLWithPath: url.path + ".meta.json")
        let metaDBURL = URL(fileURLWithPath: url.path + ".meta.db")

        try Data("stale".utf8).write(to: metaJSONURL, options: .atomic)
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: metaDBURL, options: .atomic)
        #expect(FileManager.default.fileExists(atPath: metaJSONURL.path))
        #expect(FileManager.default.fileExists(atPath: metaDBURL.path))

        try await index.save(to: url)

        #expect(!FileManager.default.fileExists(atPath: metaJSONURL.path))
        #expect(!FileManager.default.fileExists(atPath: metaDBURL.path))
    }

    @Test("Batch search returns one result list per query")
    func batchSearchReturnsCorrectShape() async throws {
        let dim = 16
        let vectors = makeVectors(count: 50, dim: dim, seedOffset: 0)
        let ids = (0..<50).map { "vec_\($0)" }
        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let queries = [vectors[1], vectors[2], vectors[3]]
        let results = try await index.batchSearch(queries: queries, k: 5)

        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.count == 5 })
    }

    @Test("Build rejects invalid node count and degree combinations")
    func buildValidationRejectsInvalidDegree() async throws {
        let singleVector = [[Float]](repeating: [0, 1, 2, 3], count: 1)
        let singleIDs = ["only"]
        let index = GraphIndex(configuration: IndexConfiguration(degree: 1, metric: .cosine))
        do {
            try await index.build(vectors: singleVector, ids: singleIDs)
            #expect(Bool(false), "Expected construction failure for single-vector build")
        } catch let error as ANNSError {
            guard case .constructionFailed = error else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
                return
            }
        }

        let vectors = makeVectors(count: 5, dim: 4, seedOffset: 1000)
        let ids = (0..<5).map { "bad-\($0)" }
        let invalidDegreeIndex = GraphIndex(configuration: IndexConfiguration(degree: 5, metric: .cosine))
        do {
            try await invalidDegreeIndex.build(vectors: vectors, ids: ids)
            #expect(Bool(false), "Expected construction failure for degree >= node count")
        } catch let error as ANNSError {
            guard case .constructionFailed = error else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
                return
            }
        }
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
