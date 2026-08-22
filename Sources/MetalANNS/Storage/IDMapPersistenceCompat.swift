import Foundation
import MetalANNSCore

extension IDMap {
    static func makeForPersistence(rows: [(String, UInt32)], nextID: UInt32) -> IDMap {
        makeForPersistence(rows: rows, numericRows: [], nextID: nextID)
    }

    static func makeForPersistence(
        rows: [(String, UInt32)],
        numericRows: [(UInt64, UInt32)],
        nextID: UInt32
    ) -> IDMap {
        struct IDMapPayload: Codable {
            let externalToInternal: [String: UInt32]
            let internalToExternal: [UInt32: String]
            let nextID: UInt32
            let numericToInternal: [UInt64: UInt32]
            let internalToNumeric: [UInt32: UInt64]
        }

        let externalToInternal = Dictionary(uniqueKeysWithValues: rows.map { ($0.0, $0.1) })
        let internalToExternal = Dictionary(uniqueKeysWithValues: rows.map { ($0.1, $0.0) })
        let numericToInternal = Dictionary(uniqueKeysWithValues: numericRows.map { ($0.0, $0.1) })
        let internalToNumeric = Dictionary(uniqueKeysWithValues: numericRows.map { ($0.1, $0.0) })
        let payload = IDMapPayload(
            externalToInternal: externalToInternal,
            internalToExternal: internalToExternal,
            nextID: nextID,
            numericToInternal: numericToInternal,
            internalToNumeric: internalToNumeric
        )

        if let data = try? JSONEncoder().encode(payload),
            let decoded = try? JSONDecoder().decode(IDMap.self, from: data)
        {
            return decoded
        }

        var fallback = IDMap()
        enum Entry {
            case string(String)
            case numeric(UInt64)
        }
        let sortedEntries = rows.map { ($0.1, Entry.string($0.0)) } + numericRows.map { ($0.1, Entry.numeric($0.0)) }
        for (_, entry) in sortedEntries.sorted(by: { $0.0 < $1.0 }) {
            switch entry {
            case .string(let externalID):
                _ = fallback.assign(externalID: externalID)
            case .numeric(let numericID):
                _ = fallback.assign(numericID: numericID)
            }
        }
        return fallback
    }
}
