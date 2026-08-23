import Foundation
import Testing

@testable import MetalANNS

@Suite("Advanced.StreamingIndex Merge Tests")
struct StreamingIndexMergeTests {
    @Test("Merge preserves all vectors")
    func mergePreservesAllVectors() async throws {
        let deltaCapacity = 10
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: deltaCapacity,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<(deltaCapacity + 5)).map { makeVector(row: $0, dim: 8) }
        for (i, vector) in vectors.enumerated() {
            try await index.insert(vector, id: "v\(i)")
        }

        #expect(await index.count == deltaCapacity + 5)

        for (i, vector) in vectors.enumerated() {
            let results = try await index.search(query: vector, k: 1)
            #expect(!results.isEmpty)
            #expect(results[0].id == "v\(i)")
        }
    }

    @Test("Background merge triggered")
    func backgroundMergeTriggered() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 8,
                mergeStrategy: .background
            ))

        for i in 0..<20 {
            try await index.insert(makeVector(row: i, dim: 8), id: "v\(i)")
        }

        try await index.flush()

        #expect(await index.count == 20)
        #expect(await index.isMerging == false)
    }

    @Test("Merge clears isMerging")
    func mergeClearsIsMerging() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 500,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<300).map { makeVector(row: $0, dim: 16) }
        let ids = (0..<300).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        // The isMerging window spans only the actor-suspension points inside
        // triggerMerge(), so catching it mid-flight races the poll task
        // against scheduler starvation on loaded CI runners (and no window
        // size fixes a task that never gets scheduled during it). The
        // observable contract is deterministic instead: not merging before,
        // fully merged with all vectors preserved after.
        #expect(await index.isMerging == false)

        try await index.flush()

        #expect(await index.isMerging == false)
        let results = try await index.search(query: vectors[0], k: 1)
        #expect(results.first?.id == "v0")
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.091) + cos(i * 0.037)
        }
    }
}
