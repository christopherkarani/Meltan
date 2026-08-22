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

        let flushDone = PollFlag()
        let flushTask = Task {
            defer { flushDone.set() }
            try await index.flush()
        }

        var sawMerging = false
        // CI runners can take minutes from flush() starting to the merge
        // actually beginning (index builds dominate), so keep polling until
        // the flush completes rather than until a wall-clock deadline. The
        // merge itself holds isMerging across several actor suspension
        // points, which a 1ms poll catches easily once it starts.
        while !flushDone.isSet {
            if await index.isMerging {
                sawMerging = true
                break
            }
            try await Task.sleep(nanoseconds: 1_000_000)
        }

        try await flushTask.value

        #expect(sawMerging)
        #expect(await index.isMerging == false)
    }

    /// Sendable completion flag shared between the flush task and the
    /// polling loop.
    private final class PollFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false

        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }

        func set() {
            lock.lock()
            defer { lock.unlock() }
            value = true
        }
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.091) + cos(i * 0.037)
        }
    }
}
