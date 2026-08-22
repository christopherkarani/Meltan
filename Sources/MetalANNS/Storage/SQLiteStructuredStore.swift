import Foundation
import MetalANNSCore
import SQLite3

enum SQLiteStructuredStore {
    private static let defaultKey = "state"
    private static let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    static func save<T: Encodable>(_ value: T, to dbURL: URL, key: String = defaultKey) throws {
        let payload = try JSONEncoder().encode(value)
        try save(data: payload, to: dbURL, key: key)
    }

    static func load<T: Decodable>(_ type: T.Type, from dbURL: URL, key: String = defaultKey) throws -> T? {
        guard let payload = try loadData(from: dbURL, key: key) else {
            return nil
        }
        return try JSONDecoder().decode(T.self, from: payload)
    }

    private static func save(data: Data, to dbURL: URL, key: String) throws {
        let fileManager = FileManager.default
        let parentURL = dbURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentURL, withIntermediateDirectories: true)

        let tempURL = parentURL.appendingPathComponent(".\(dbURL.lastPathComponent).tmp-\(UUID().uuidString)")
        try writeData(data, key: key, to: tempURL)

        if fileManager.fileExists(atPath: dbURL.path) {
            _ = try fileManager.replaceItemAt(dbURL, withItemAt: tempURL)
        } else {
            try fileManager.moveItem(at: tempURL, to: dbURL)
        }
    }

    private static func loadData(from dbURL: URL, key: String) throws -> Data? {
        guard FileManager.default.fileExists(atPath: dbURL.path) else {
            return nil
        }

        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            defer { if db != nil { sqlite3_close(db) } }
            throw ANNSError.corruptFile("Failed to open SQLite metadata store: \(sqliteErrorMessage(db))")
        }
        defer { sqlite3_close(db) }

        let querySQL = "SELECT value FROM kv WHERE key = ?1"
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, querySQL, -1, &statement, nil) == SQLITE_OK else {
            throw ANNSError.corruptFile("Failed to prepare SQLite metadata read: \(sqliteErrorMessage(db))")
        }
        defer { sqlite3_finalize(statement) }

        sqlite3_bind_text(statement, 1, key, -1, sqliteTransient)

        let step = sqlite3_step(statement)
        guard step == SQLITE_ROW else {
            if step == SQLITE_DONE {
                return nil
            }
            throw ANNSError.corruptFile("Failed to read SQLite metadata row: \(sqliteErrorMessage(db))")
        }

        guard let rawBlob = sqlite3_column_blob(statement, 0) else {
            throw ANNSError.corruptFile("SQLite metadata row is missing payload")
        }
        let blobLength = Int(sqlite3_column_bytes(statement, 0))
        return Data(bytes: rawBlob, count: max(blobLength, 0))
    }

    private static func writeData(_ payload: Data, key: String, to dbURL: URL) throws {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil) == SQLITE_OK else {
            defer { if db != nil { sqlite3_close(db) } }
            throw ANNSError.constructionFailed("Failed to create SQLite metadata store: \(sqliteErrorMessage(db))")
        }
        defer { sqlite3_close(db) }

        try executeSQL("PRAGMA journal_mode=DELETE", on: db)
        try executeSQL("PRAGMA synchronous=FULL", on: db)
        try executeSQL("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value BLOB NOT NULL)", on: db)

        let upsertSQL = "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)"
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, upsertSQL, -1, &statement, nil) == SQLITE_OK else {
            throw ANNSError.constructionFailed("Failed to prepare SQLite metadata write: \(sqliteErrorMessage(db))")
        }
        defer { sqlite3_finalize(statement) }

        sqlite3_bind_text(statement, 1, key, -1, sqliteTransient)
        payload.withUnsafeBytes { rawBuffer in
            sqlite3_bind_blob(statement, 2, rawBuffer.baseAddress, Int32(rawBuffer.count), sqliteTransient)
        }

        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw ANNSError.constructionFailed("Failed to write SQLite metadata row: \(sqliteErrorMessage(db))")
        }
    }

    private static func executeSQL(_ sql: String, on db: OpaquePointer?) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else {
            throw ANNSError.constructionFailed("SQLite exec failed for '\(sql)': \(sqliteErrorMessage(db))")
        }
    }

    private static func sqliteErrorMessage(_ db: OpaquePointer?) -> String {
        if let cString = sqlite3_errmsg(db) {
            return String(cString: cString)
        }
        return "Unknown SQLite error"
    }
}
