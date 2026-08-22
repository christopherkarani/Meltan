import Foundation
import Testing

@testable import MetalANNS

@Suite("Advanced.StreamingIndex Flush Tests")
struct StreamingIndexFlushTests {
    @Test("Flush merges all pending")
    func flushMergesAllPending() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 30,
                mergeStrategy: .blocking
            ))

        for i in 0..<50 {
            try await index.insert(makeVector(row: i, dim: 16), id: "v\(i)")
        }

        try await index.flush()

        #expect(await index.count == 50)
        #expect(await index.isMerging == false)
    }

    @Test("Flush idempotent")
    func flushIdempotent() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 30,
                mergeStrategy: .blocking
            ))

        for i in 0..<20 {
            try await index.insert(makeVector(row: i, dim: 16), id: "v\(i)")
        }

        try await index.flush()
        try await index.flush()

        #expect(await index.count == 20)
        #expect(await index.isMerging == false)
    }

    @Test("Concurrent insert and search")
    func concurrentInsertAndSearch() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 15,
                mergeStrategy: .background
            ))

        let seedVectors = (0..<10).map { makeVector(row: $0, dim: 16) }
        let seedIDs = (0..<10).map { "seed-\($0)" }
        try await index.batchInsert(seedVectors, ids: seedIDs)

        let insertVectorsA = (0..<20).map { makeVector(row: 10_000 + $0, dim: 16) }
        let insertIDsA = (0..<20).map { "a-\($0)" }
        let insertVectorsB = (0..<20).map { makeVector(row: 20_000 + $0, dim: 16) }
        let insertIDsB = (0..<20).map { "b-\($0)" }

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await index.batchInsert(insertVectorsA, ids: insertIDsA)
            }
            group.addTask {
                try await index.batchInsert(insertVectorsB, ids: insertIDsB)
            }
            group.addTask {
                _ = try await index.search(query: seedVectors[0], k: 10)
            }
            group.addTask {
                _ = try await index.search(query: seedVectors[1], k: 10)
            }
            for try await _ in group {}
        }

        try await index.flush()
        #expect(await index.count >= 30)
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.081) + cos(i * 0.047)
        }
    }
}
