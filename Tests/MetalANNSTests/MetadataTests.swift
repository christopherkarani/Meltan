import Testing

@testable import MetalANNSCore

@Suite("Metadata & IDMap Tests")
struct MetadataTests {
    @Test("MetadataBuffer roundtrip")
    func metadataRoundtrip() throws {
        let metadata = try MetadataBuffer()

        metadata.entryPointID = 42
        metadata.nodeCount = 1000
        metadata.degree = 32
        metadata.dim = 128
        metadata.iterationCount = 7

        #expect(metadata.entryPointID == 42)
        #expect(metadata.nodeCount == 1000)
        #expect(metadata.degree == 32)
        #expect(metadata.dim == 128)
        #expect(metadata.iterationCount == 7)
    }

    @Test("IDMap bidirectional mapping")
    func idMapMapping() {
        var idMap = IDMap()

        let idA = idMap.assign(externalID: "doc-a")
        let idB = idMap.assign(externalID: "doc-b")

        #expect(idA == 0)
        #expect(idB == 1)
        #expect(idMap.internalID(for: "doc-a") == 0)
        #expect(idMap.internalID(for: "doc-b") == 1)
        #expect(idMap.externalID(for: 0) == "doc-a")
        #expect(idMap.externalID(for: 1) == "doc-b")
        #expect(idMap.count == 2)
    }

    @Test("IDMap rejects duplicate external IDs")
    func idMapDuplicate() {
        var idMap = IDMap()

        let first = idMap.assign(externalID: "doc-a")
        let second = idMap.assign(externalID: "doc-a")

        #expect(first == 0)
        #expect(second == nil)
        #expect(idMap.count == 1)
    }
}
