import Foundation
import GRDB
import MetalANNSCore

/// SQLite-backed persistence layer for index structured data.
// GRDB handles internal concurrency; this wrapper is immutable after init.
public final class IndexDatabase: @unchecked Sendable {
    public let pool: DatabasePool

    public init(path: String) throws {
        var configuration = Configuration()
        configuration.prepareDatabase { db in
            try db.execute(sql: "PRAGMA journal_mode = WAL")
            try db.execute(sql: "PRAGMA mmap_size = 268435456")
        }
        self.pool = try DatabasePool(path: path, configuration: configuration)
        try migrate()
    }

    public func prepareForFileMove() throws {
        try pool.writeWithoutTransaction { db in
            try db.execute(sql: "PRAGMA wal_checkpoint(TRUNCATE)")
        }
    }

    private func migrate() throws {
        var migrator = DatabaseMigrator()
        migrator.registerMigration("v1-foundation") { db in
            try db.create(table: "idmap") { table in
                table.column("externalID", .text).notNull().primaryKey()
                table.column("internalID", .integer).notNull().unique()
            }
            try db.create(index: "idmap_by_internal", on: "idmap", columns: ["internalID"])

            try db.create(table: "config") { table in
                table.column("key", .text).notNull().primaryKey()
                table.column("value", .text).notNull()
            }

            try db.create(table: "soft_deletion") { table in
                table.column("internalID", .integer).notNull().primaryKey()
            }
        }
        migrator.registerMigration("v2-idmap-numeric") { db in
            try db.create(table: "idmap_numeric") { table in
                // Store as text to preserve the full UInt64 domain.
                table.column("numericID", .text).notNull().primaryKey()
                table.column("internalID", .integer).notNull().unique()
            }
            try db.create(index: "idmap_numeric_by_internal", on: "idmap_numeric", columns: ["internalID"])
        }
        try migrator.migrate(pool)
    }
}

// MARK: - IDMap persistence

extension IndexDatabase {
    public func saveIDMap(_ idMap: IDMap) throws {
        let stringRows = (0..<idMap.nextInternalID).compactMap { index -> (String, UInt32)? in
            let internalID = UInt32(index)
            guard let externalID = idMap.externalID(for: internalID) else {
                return nil
            }
            return (externalID, internalID)
        }
        let numericRows = (0..<idMap.nextInternalID).compactMap { index -> (String, UInt32)? in
            let internalID = UInt32(index)
            guard let numericID = idMap.numericID(for: internalID) else {
                return nil
            }
            return (String(numericID), internalID)
        }

        try pool.write { db in
            try db.execute(sql: "DELETE FROM idmap")
            try db.execute(sql: "DELETE FROM idmap_numeric")
            let statement = try db.makeStatement(sql: "INSERT INTO idmap (externalID, internalID) VALUES (?, ?)")
            let numericStatement = try db.makeStatement(
                sql: "INSERT INTO idmap_numeric (numericID, internalID) VALUES (?, ?)"
            )

            for (externalID, internalID) in stringRows {
                try statement.execute(arguments: [externalID, Int64(internalID)])
            }
            for (numericID, internalID) in numericRows {
                try numericStatement.execute(arguments: [numericID, Int64(internalID)])
            }

            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["idmap.nextID", String(idMap.nextInternalID)]
            )
        }
    }

    public func loadIDMap() throws -> IDMap {
        try pool.read { db in
            let stringRows = try Row.fetchAll(db, sql: "SELECT externalID, internalID FROM idmap ORDER BY internalID")
            var stringEntries: [(String, UInt32)] = []
            stringEntries.reserveCapacity(stringRows.count)
            for row in stringRows {
                let externalID: String = row["externalID"]
                let internalID: Int64 = row["internalID"]
                guard let internalIDUInt32 = UInt32(exactly: internalID) else {
                    throw ANNSError.corruptFile("Invalid idmap internalID value: \(internalID)")
                }
                stringEntries.append((externalID, internalIDUInt32))
            }

            let numericRows = try Row.fetchAll(
                db,
                sql: "SELECT numericID, internalID FROM idmap_numeric ORDER BY internalID"
            )
            var numericEntries: [(UInt64, UInt32)] = []
            numericEntries.reserveCapacity(numericRows.count)
            for row in numericRows {
                let numericIDString: String = row["numericID"]
                let internalID: Int64 = row["internalID"]
                guard let numericID = UInt64(numericIDString) else {
                    throw ANNSError.corruptFile("Invalid idmap_numeric numericID value: \(numericIDString)")
                }
                guard let internalIDUInt32 = UInt32(exactly: internalID) else {
                    throw ANNSError.corruptFile("Invalid idmap_numeric internalID value: \(internalID)")
                }
                numericEntries.append((numericID, internalIDUInt32))
            }

            let nextIDString = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["idmap.nextID"]
            )
            let nextID: UInt32
            if let nextIDString {
                guard let parsedNextID = UInt32(nextIDString) else {
                    throw ANNSError.corruptFile("Invalid idmap.nextID value: \(nextIDString)")
                }
                nextID = parsedNextID
            } else {
                nextID = UInt32(stringEntries.count + numericEntries.count)
            }

            return IDMap.makeForPersistence(rows: stringEntries, numericRows: numericEntries, nextID: nextID)
        }
    }
}

// MARK: - Configuration persistence

extension IndexDatabase {
    public func saveConfiguration(_ configuration: IndexConfiguration) throws {
        let data = try JSONEncoder().encode(configuration)
        guard let encoded = String(data: data, encoding: .utf8) else {
            throw ANNSError.constructionFailed("Failed to encode configuration")
        }

        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["index.configuration", encoded]
            )
        }
    }

    public func loadConfiguration() throws -> IndexConfiguration? {
        try pool.read { db in
            guard
                let payload = try String.fetchOne(
                    db,
                    sql: "SELECT value FROM config WHERE key = ?",
                    arguments: ["index.configuration"]
                ), let data = payload.data(using: .utf8)
            else {
                return nil
            }
            return try JSONDecoder().decode(IndexConfiguration.self, from: data)
        }
    }
}

// MARK: - Soft-deletion persistence

extension IndexDatabase {
    public func saveSoftDeletion(_ softDeletion: SoftDeletion) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM soft_deletion")

            for internalID in softDeletion.allDeletedIDs {
                try db.execute(
                    sql: "INSERT INTO soft_deletion (internalID) VALUES (?)",
                    arguments: [Int64(internalID)]
                )
            }
        }
    }

    public func loadSoftDeletion() throws -> SoftDeletion {
        try pool.read { db in
            let values = try Int64.fetchAll(db, sql: "SELECT internalID FROM soft_deletion")
            var softDeletion = SoftDeletion()
            for value in values {
                guard let internalID = UInt32(exactly: value) else {
                    throw ANNSError.corruptFile("Invalid soft_deletion internalID value: \(value)")
                }
                softDeletion.markDeleted(internalID)
            }
            return softDeletion
        }
    }
}

// MARK: - Metadata persistence

extension IndexDatabase {
    public func saveMetadataStore(_ metadataStore: MetadataStore) throws {
        let data = try JSONEncoder().encode(metadataStore)
        guard let encoded = String(data: data, encoding: .utf8) else {
            throw ANNSError.constructionFailed("Failed to encode metadataStore")
        }

        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["index.metadataStore", encoded]
            )
        }
    }

    public func loadMetadataStore() throws -> MetadataStore {
        try pool.read { db in
            guard
                let payload = try String.fetchOne(
                    db,
                    sql: "SELECT value FROM config WHERE key = ?",
                    arguments: ["index.metadataStore"]
                ), let data = payload.data(using: .utf8)
            else {
                return MetadataStore()
            }
            return try JSONDecoder().decode(MetadataStore.self, from: data)
        }
    }
}
