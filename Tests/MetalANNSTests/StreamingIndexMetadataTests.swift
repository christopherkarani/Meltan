import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Advanced.StreamingIndex Metadata Tests")
struct StreamingIndexMetadataTests {
    @Test("Set and get string metadata")
    func setAndGetStringMetadata() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 100,
                mergeStrategy: .blocking
            ))
        let vector = makeVector(row: 0, dim: 16)

        try await index.insert(vector, id: "v0")
        try await index.insert(makeVector(row: 1, dim: 16), id: "v1")
        try await index.setMetadata("tag", value: "hot", for: "v0")

        let results = try await index.search(
            query: vector,
            k: 10,
            filter: .equals(column: "tag", value: "hot")
        )

        #expect(results.contains(where: { $0.id == "v0" }))
    }

    @Test("Metadata preserved after merge")
    func metadataPreservedAfterMerge() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 10,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<20).map { makeVector(row: $0, dim: 16) }
        let ids = (0..<20).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.setMetadata("tag", value: "hot", for: "v3")

        try await index.flush()

        let results = try await index.search(
            query: vectors[3],
            k: 10,
            filter: .equals(column: "tag", value: "hot")
        )

        #expect(results.contains(where: { $0.id == "v3" }))
    }

    @Test("Delete removes from results")
    func deleteRemovesFromResults() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 20,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<10).map { makeVector(row: 100 + $0, dim: 16) }
        let ids = (0..<10).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.delete(id: "v5")

        let results = try await index.search(query: vectors[4], k: 20)
        #expect(!results.contains(where: { $0.id == "v5" }))
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.113) + cos(i * 0.057)
        }
    }
}
