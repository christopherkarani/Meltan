import Foundation
import Testing

@testable import MetalANNSCore

@Suite("IDMap Tests")
struct IDMapTests {
    @Test func assignUInt64Key() {
        var map = IDMap()
        let internalID = map.assign(numericID: 42)
        #expect(internalID != nil)
        #expect(map.internalID(forNumeric: 42) == internalID)
        #expect(map.numericID(for: internalID!) == 42)
    }

    @Test func uint64KeyAndStringKeyAreIndependentNamespaces() {
        var map = IDMap()
        // String "42" and UInt64 42 occupy separate namespaces but share
        // the same internal ID counter — so they get consecutive internal IDs.
        let strInternalID = map.assign(externalID: "42")
        let numInternalID = map.assign(numericID: 42)
        #expect(strInternalID != nil)
        #expect(numInternalID != nil)
        #expect(strInternalID != numInternalID, "Share counter but different slots")
        // Lookup in the wrong namespace returns nil.
        #expect(map.internalID(for: "42") == strInternalID)
        #expect(map.internalID(forNumeric: 42) == numInternalID)
        #expect(map.externalID(for: numInternalID!) == nil, "UInt64 slot has no String ID")
        #expect(map.numericID(for: strInternalID!) == nil, "String slot has no UInt64 ID")
    }

    @Test func uint64EdgeValues() {
        var map = IDMap()
        let edgeCases: [UInt64] = [0, 1, UInt64.max - 1, 12_345_678_901_234]
        for val in edgeCases {
            let internalID = map.assign(numericID: val)
            #expect(internalID != nil, "assign(\(val)) returned nil")
            #expect(map.numericID(for: internalID!) == val, "round-trip failed for \(val)")
        }
    }

    @Test func duplicateUInt64KeyReturnsNil() {
        var map = IDMap()
        let first = map.assign(numericID: 99)
        let second = map.assign(numericID: 99)
        #expect(first != nil)
        #expect(second == nil, "Duplicate assign must return nil")
    }

    @Test func countIncludesBothStringAndUInt64Keys() {
        var map = IDMap()
        _ = map.assign(externalID: "a")
        _ = map.assign(externalID: "b")
        _ = map.assign(numericID: 1)
        _ = map.assign(numericID: 2)
        #expect(map.count == 4)
    }

    @Test func canAllocateReflectsSharedCounter() {
        var map = IDMap()
        // Insert 3 via String, 3 via UInt64 = 6 total internal IDs consumed.
        for i in 0..<3 { _ = map.assign(externalID: "s\(i)") }
        for i in 0..<3 { _ = map.assign(numericID: UInt64(i)) }
        #expect(map.count == 6)
        #expect(map.canAllocate(1))
    }

    @Test func decodeLegacyPayloadWithoutNumericMaps() throws {
        struct LegacyPayload: Codable {
            let externalToInternal: [String: UInt32]
            let internalToExternal: [UInt32: String]
            let nextID: UInt32
        }

        let data = try JSONEncoder().encode(
            LegacyPayload(
                externalToInternal: ["legacy": 0],
                internalToExternal: [0: "legacy"],
                nextID: 1
            )
        )
        let decoded = try JSONDecoder().decode(IDMap.self, from: data)
        #expect(decoded.internalID(for: "legacy") == 0)
        #expect(decoded.externalID(for: 0) == "legacy")
        #expect(decoded.internalID(forNumeric: 0) == nil)
        #expect(decoded.numericID(for: 0) == nil)
        #expect(decoded.count == 1)
    }
}
