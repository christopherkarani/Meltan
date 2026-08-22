import Foundation
import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Bounded Selection Tests")
struct BoundedSelectionTests {
    // MARK: - Primitive pins

    @Test("Binary heap drains best-first")
    func binaryHeapDrainsBestFirst() {
        var heap = BinaryHeap<Int> { lhs, rhs in lhs < rhs }
        for value in [5, 1, 4, 2, 3] {
            heap.push(value)
        }
        var drained: [Int] = []
        while let value = heap.pop() {
            drained.append(value)
        }
        #expect(drained == [1, 2, 3, 4, 5])
    }

    @Test("Bounded buffer rejects boundary ties when full")
    func boundedBufferRejectsBoundaryTies() {
        var buffer = BoundedPriorityBuffer<Float>(capacity: 2) { $0 < $1 }
        buffer.insert(7)
        buffer.insert(4)
        buffer.insert(7)
        #expect(buffer.count == 2)
        #expect(buffer.sortedElements() == [4, 7])
    }

    @Test("Bounded buffer with zero capacity stores nothing")
    func boundedBufferZeroCapacity() {
        var buffer = BoundedPriorityBuffer<Int>(capacity: 0) { $0 < $1 }
        buffer.insert(1)
        buffer.insert(2)
        #expect(buffer.count == 0)
        #expect(buffer.sortedElements() == [])
    }

    @Test("Bounded buffer matches sort-and-prefix reference on seeded streams")
    func boundedBufferMatchesReference() {
        var rng = SeededGenerator(state: 2026)
        for trial in 0..<24 {
            let length = Int.random(in: 1...160, using: &rng)
            let capacity = Int.random(in: 1...32, using: &rng)
            var stream = (0..<length).map { _ in Float.random(in: -10...10, using: &rng) }
            if trial % 2 == 0 {
                stream = stream.map { ($0 * 4).rounded(.toNearestOrAwayFromZero) / 4 }
            }

            var buffer = BoundedPriorityBuffer<Float>(capacity: capacity) { $0 < $1 }
            for value in stream {
                buffer.insert(value)
            }
            let expected = Array(stream.sorted(by: <).prefix(capacity))
            #expect(buffer.sortedElements() == expected, "trial \(trial)")
        }
    }

    // MARK: - Shared fixtures

    private func makeContext() throws -> MetalContext {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        return try MetalContext()
    }

    private func duplicatedCorpus(dim: Int, pairs: Int, seed: UInt64) -> [[Float]] {
        var rng = SeededGenerator(state: seed)
        let distinct = (0..<pairs).map { _ in (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) } }
        var rows: [[Float]] = []
        for vector in distinct {
            rows.append(vector)
            rows.append(vector)
        }
        return rows
    }

    private func makeBuffer(_ vectors: [[Float]], dim: Int, device: any MTLDevice) throws -> VectorBuffer {
        let buffer = try VectorBuffer(capacity: vectors.count, dim: dim, device: device)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(vectors.count)
        return buffer
    }

    private func pinnedQuery(dim: Int) -> [Float] {
        var rng = SeededGenerator(state: 99)
        return (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    private func assertScoresAscending(_ results: [SearchResult]) {
        guard results.count > 1 else {
            Issue.record("Expected multiple results")
            return
        }
        for index in 1..<results.count {
            #expect(results[index - 1].score <= results[index].score, "scores must ascend")
        }
    }

    // MARK: - Host scan seam pins

    @Test("Host flat scan keeps pinned tie order under duplicate vectors")
    func hostScanTieOrderPinned() throws {
        let context = try makeContext()
        let dim = 8
        let vectors = duplicatedCorpus(dim: dim, pairs: 6, seed: 11)
        let buffer = try makeBuffer(vectors, dim: dim, device: context.device)
        let query = pinnedQuery(dim: dim)

        let cosine = FlatGPUSearch.hostSearch(query: query, vectors: buffer, k: 5, metric: .cosine)
        #expect(cosine.map(\.internalID) == [4, 5, 7, 6, 0])
        assertScoresAscending(cosine)

        let full = FlatGPUSearch.hostSearch(query: query, vectors: buffer, k: 12, metric: .cosine)
        #expect(full.map(\.internalID) == [4, 5, 7, 6, 0, 1, 2, 3, 9, 8, 11, 10])

        let clamped = FlatGPUSearch.hostSearch(query: query, vectors: buffer, k: 20, metric: .cosine)
        #expect(clamped.map(\.internalID) == [4, 5, 7, 6, 0, 1, 2, 3, 9, 8, 11, 10])

        let l2 = FlatGPUSearch.hostSearch(query: query, vectors: buffer, k: 5, metric: .l2)
        #expect(l2.map(\.internalID) == [4, 5, 7, 6, 0])
        assertScoresAscending(l2)

        let innerProduct = FlatGPUSearch.hostSearch(query: query, vectors: buffer, k: 5, metric: .innerProduct)
        #expect(innerProduct.map(\.internalID) == [6, 7, 5, 4, 0])
        assertScoresAscending(innerProduct)
    }

    // MARK: - GPU tier seam pins

    @Test("GPU flat scan tier keeps pinned tie order under duplicate vectors")
    func gpuTierTieOrderPinned() async throws {
        let context = try makeContext()
        let dim = 8
        let vectors = duplicatedCorpus(dim: dim, pairs: 6, seed: 11)
        let buffer = try makeBuffer(vectors, dim: dim, device: context.device)
        let query = pinnedQuery(dim: dim)

        let cosine = try await FlatGPUSearch.batchSearch(
            context: context, queries: [query], vectors: buffer, k: 5, metric: .cosine,
            tierOverride: vectors.count - 1)
        // Post-#12 GPU tier selection orders duplicate-distance ties by
        // ascending id (boundedTopK tie sort), unlike the host scan tier.
        #expect(cosine[0].map(\.internalID) == [4, 5, 6, 7, 0])
        assertScoresAscending(cosine[0])

        let l2 = try await FlatGPUSearch.batchSearch(
            context: context, queries: [query], vectors: buffer, k: 7, metric: .l2,
            tierOverride: vectors.count - 1)
        #expect(l2[0].map(\.internalID) == [4, 5, 6, 7, 0, 1, 3])
    }

    // MARK: - IVFPQ merge seam pin

    @Test("IVFPQ search merges tied scores newest-first")
    func ivfpqMergeTieOrderPinned() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 4,
            numCentroids: 256,
            numCoarseCentroids: 4,
            nprobe: 4,
            metric: .l2,
            trainingIterations: 4
        )
        let index = try Advanced.IVFPQIndex(capacity: 256, dimension: 16, config: config)

        var rng = SeededGenerator(state: 3)
        let training = (0..<300).map { _ in (0..<16).map { _ in Float.random(in: -1...1, using: &rng) } }
        try await index.train(vectors: training)

        var databaseRng = SeededGenerator(state: 5)
        let distinct = (0..<10).map { _ in (0..<16).map { _ in Float.random(in: -1...1, using: &databaseRng) } }
        var vectors: [[Float]] = []
        var ids: [String] = []
        for (position, vector) in distinct.enumerated() {
            vectors.append(vector)
            ids.append("a\(position)")
            vectors.append(vector)
            ids.append("b\(position)")
        }
        try await index.add(vectors: vectors, ids: ids)

        let results = await index.search(query: vectors[0], k: 6)
        #expect(results.map(\.id) == ["b0", "a0", "b2", "a2", "b4", "a4"])
        for pair in stride(from: 0, to: results.count - 1, by: 2) {
            #expect(results[pair].score == results[pair + 1].score)
            if pair + 2 < results.count {
                #expect(results[pair + 1].score < results[pair + 2].score)
            }
        }
    }
}
