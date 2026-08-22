import Foundation
import GRDB
import MetalANNSCore

/// SQLite-backed persistence for _StreamingIndex state.
///
/// This stores vectors as per-row BLOBs and keeps streaming metadata keyed by
/// external string IDs used by _StreamingIndex.
public final class StreamingDatabase: Sendable {
    let pool: DatabasePool

    public init(path: String) throws {
        var config = Configuration()
        config.prepareDatabase { db in
            try db.execute(sql: "PRAGMA mmap_size = 268435456")
        }
        pool = try DatabasePool(path: path, configuration: config)
        try migrate()
    }

    private func migrate() throws {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1-streaming") { db in
            try db.create(table: "vectors") { t in
                t.autoIncrementedPrimaryKey("rowID")
                t.column("externalID", .text).notNull().unique()
                t.column("data", .blob).notNull()
            }

            try db.create(table: "deleted") { t in
                t.column("externalID", .text).notNull().primaryKey()
            }

            try db.create(table: "vector_metadata") { t in
                t.column("externalID", .text).notNull()
                t.column("key", .text).notNull()
                t.column("value", .text).notNull()
                t.primaryKey(["externalID", "key"])
            }

            try db.create(table: "config") { t in
                t.column("key", .text).notNull().primaryKey()
                t.column("value", .text).notNull()
            }

            try db.create(table: "state") { t in
                t.column("key", .text).notNull().primaryKey()
                t.column("intValue", .integer)
            }
        }

        try migrator.migrate(pool)
    }

    // MARK: - Vectors

    public func insertVectors(_ vectors: [[Float]], ids: [String]) throws {
        precondition(vectors.count == ids.count)
        try pool.write { db in
            let statement = try db.makeStatement(
                sql: "INSERT OR REPLACE INTO vectors (externalID, data) VALUES (?, ?)"
            )
            for (vector, id) in zip(vectors, ids) {
                let data = vector.withUnsafeBytes { Data($0) }
                try statement.execute(arguments: [id, data])
            }
        }
    }

    public func loadAllVectors() throws -> (vectors: [[Float]], ids: [String]) {
        try pool.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: "SELECT externalID, data FROM vectors ORDER BY rowID"
            )

            var vectors: [[Float]] = []
            var ids: [String] = []
            vectors.reserveCapacity(rows.count)
            ids.reserveCapacity(rows.count)

            for row in rows {
                let id: String = row["externalID"]
                let data: Data = row["data"]
                let vector = try decodeVectorBlob(data)
                ids.append(id)
                vectors.append(vector)
            }
            return (vectors, ids)
        }
    }

    // MARK: - Deletion

    public func markDeleted(ids: Set<String>) throws {
        try pool.write { db in
            let statement = try db.makeStatement(
                sql: "INSERT OR IGNORE INTO deleted (externalID) VALUES (?)"
            )
            for id in ids {
                try statement.execute(arguments: [id])
            }
        }
    }

    public func loadDeletedIDs() throws -> Set<String> {
        try pool.read { db in
            Set(try String.fetchAll(db, sql: "SELECT externalID FROM deleted"))
        }
    }

    // MARK: - Config

    public func saveConfig(_ config: StreamingConfiguration) throws {
        let data = try JSONEncoder().encode(config)
        guard let json = String(data: data, encoding: .utf8) else {
            throw ANNSError.constructionFailed("Failed to encode streaming config")
        }

        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["streaming.config", json]
            )
        }
    }

    public func loadConfig() throws -> StreamingConfiguration? {
        try pool.read { db in
            guard
                let json = try String.fetchOne(
                    db,
                    sql: "SELECT value FROM config WHERE key = ?",
                    arguments: ["streaming.config"]
                )
            else {
                return nil
            }

            return try JSONDecoder().decode(StreamingConfiguration.self, from: Data(json.utf8))
        }
    }

    // MARK: - Per-vector metadata

    public func saveVectorMetadata(id: String, metadata: [String: String]) throws {
        try pool.write { db in
            try db.execute(
                sql: "DELETE FROM vector_metadata WHERE externalID = ?",
                arguments: [id]
            )

            let statement = try db.makeStatement(
                sql: "INSERT INTO vector_metadata (externalID, key, value) VALUES (?, ?, ?)"
            )
            for (key, value) in metadata {
                try statement.execute(arguments: [id, key, value])
            }
        }
    }

    public func loadVectorMetadata(id: String) throws -> [String: String]? {
        try pool.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: "SELECT key, value FROM vector_metadata WHERE externalID = ?",
                arguments: [id]
            )
            guard !rows.isEmpty else {
                return nil
            }

            var metadata: [String: String] = [:]
            metadata.reserveCapacity(rows.count)
            for row in rows {
                let key: String = row["key"]
                let value: String = row["value"]
                metadata[key] = value
            }
            return metadata
        }
    }

    public func saveAllVectorMetadata(_ metadataByID: [String: [String: String]]) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM vector_metadata")
            let statement = try db.makeStatement(
                sql: "INSERT INTO vector_metadata (externalID, key, value) VALUES (?, ?, ?)"
            )

            for (id, metadata) in metadataByID {
                for (key, value) in metadata {
                    try statement.execute(arguments: [id, key, value])
                }
            }
        }
    }

    public func loadAllVectorMetadata() throws -> [String: [String: String]] {
        try pool.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: "SELECT externalID, key, value FROM vector_metadata"
            )

            var metadataByID: [String: [String: String]] = [:]
            for row in rows {
                let id: String = row["externalID"]
                let key: String = row["key"]
                let value: String = row["value"]
                metadataByID[id, default: [:]][key] = value
            }
            return metadataByID
        }
    }

    // MARK: - State

    public func saveVectorDimension(_ dimension: Int) throws {
        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO state (key, intValue) VALUES (?, ?)",
                arguments: ["vectorDimension", dimension]
            )
        }
    }

    public func loadVectorDimension() throws -> Int? {
        try pool.read { db in
            try Int.fetchOne(db, sql: "SELECT intValue FROM state WHERE key = ?", arguments: ["vectorDimension"])
        }
    }

    // MARK: - Internals

    private func decodeVectorBlob(_ data: Data) throws -> [Float] {
        let scalarSize = MemoryLayout<Float>.size
        guard data.count % scalarSize == 0 else {
            throw ANNSError.corruptFile("Invalid streaming vector blob size: \(data.count)")
        }

        let count = data.count / scalarSize
        var vector = Array(repeating: Float.zero, count: count)
        _ = vector.withUnsafeMutableBytes { buffer in
            data.copyBytes(to: buffer)
        }
        return vector
    }
}
