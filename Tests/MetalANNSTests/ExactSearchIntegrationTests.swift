import Foundation
import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

/// Deterministic regression coverage for the fused exact-search path:
/// - GPU-tier kernel exactness (tier boundary forced below corpus size)
/// - Fallback to graph traversal when the exact path is disabled or ineligible
/// - Exactness preserved through inserts, deletes, compaction, save/load
@Suite("Exact Search Integration Tests")
struct ExactSearchIntegrationTests {
    // MARK: - Helpers

    private func makeContext() throws -> MetalContext {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        return try MetalContext()
    }

    private func seededVectors(count: Int, dim: Int, seed: UInt64 = 7) -> [[Float]] {
        var state = seed
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2.0 - 1.0
        }
        return (0..<count).map { _ in (0..<dim).map { _ in next() } }
    }

    private func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric,
        skipIndices: Set<Int> = []
    ) -> Set<UInt32> {
        let scored = vectors.enumerated().compactMap { index, vector -> (UInt32, Float)? in
            guard !skipIndices.contains(index) else { return nil }
            return (UInt32(index), distance(query: query, vector: vector, metric: metric))
        }
        return Set(
            scored.sorted { lhs, rhs in
                lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
            }
            .prefix(k)
            .map(\.0))
    }

    private func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        case .hamming:
            return 0
        }
    }

    /// Runs body with an explicit tier override so the Metal flat-scan kernel
    /// executes even on small corpora. The override is passed per call — no
    /// shared static state is mutated, so parallel test suites never see it.
    private func withGPUForceTier<T>(corpusSize: Int, _ body: (Int) async throws -> T) async throws -> T {
        try await body(corpusSize - 1)
    }

    // MARK: - GPU tier kernel correctness

    @Test("GPU tier scan matches brute force across metrics")
    func gpuTierMatchesBruteForce() async throws {
        let context = try makeContextGuarded()
        try await withGPUForceTier(corpusSize: 2048) { tierOverride in
            let dimCases = [64, 66]
            for dim in dimCases {
                let count = 2048
                let vectors = seededVectors(count: count, dim: dim, seed: UInt64(dim) &+ 100)
                let buffer = try VectorBuffer(capacity: count, dim: dim, device: context.device)
                try buffer.batchInsert(vectors: vectors, startingAt: 0)
                buffer.setCount(count)

                for metric in [Metric.cosine, .l2, .innerProduct] {
                    let queries = seededVectors(count: 4, dim: dim, seed: UInt64(dim) &+ 777)
                    let results = try await FlatGPUSearch.batchSearch(
                        context: context, queries: queries, vectors: buffer, k: 50, metric: metric,
                        tierOverride: tierOverride)
                    #expect(results.count == queries.count)
                    for (queryIndex, resultRow) in results.enumerated() {
                        let expected = bruteForceTopK(
                            query: queries[queryIndex], vectors: vectors, k: 50, metric: metric)
                        #expect(
                            Set(resultRow.map(\.internalID)) == expected,
                            "GPU tier mismatch dim=\(dim) metric=\(metric) query=\(queryIndex)")
                    }
                }
            }
        }
    }

    @Test("GPU tier batch chunking stays exact across dispatch boundaries")
    func gpuTierBatchChunking() async throws {
        let context = try makeContextGuarded()
        try await withGPUForceTier(corpusSize: 1024) { tierOverride in
            let count = 1024
            let dim = 32
            let vectors = seededVectors(count: count, dim: dim, seed: 404)
            let buffer = try VectorBuffer(capacity: count, dim: dim, device: context.device)
            try buffer.batchInsert(vectors: vectors, startingAt: 0)
            buffer.setCount(count)

            let queries = seededVectors(count: 9, dim: dim, seed: 808)
            let batches = try await FlatGPUSearch.batchSearch(
                context: context, queries: queries, vectors: buffer, k: 20, metric: .cosine,
                tierOverride: tierOverride)
            #expect(batches.count == queries.count)
            for (queryIndex, row) in batches.enumerated() {
                let expected = bruteForceTopK(
                    query: queries[queryIndex], vectors: vectors, k: 20, metric: .cosine)
                #expect(Set(row.map(\.internalID)) == expected, "chunked batch mismatch q=\(queryIndex)")
            }
        }
    }

    /// Regression: chunked parallel top-K selection must keep candidate ids
    /// global (chunk-local ids silently returned wrong rows above the
    /// parallel-selection threshold).
    @Test("Parallel GPU-tier selection stays exact past the fan-out threshold")
    func parallelTierSelectionStaysExact() async throws {
        let context = try makeContextGuarded()
        let threshold = FlatGPUSearch.parallelSelectionMinVectorCount
        // Enough rows for ≥2 lanes plus a non-divisible remainder chunk.
        let count = threshold * 2 + 123
        let dim = 8

        var vectors = seededVectors(count: count, dim: dim, seed: 4_242)
        // Duplicate a run of rows so ties exercise deterministic merge order,
        // and place them where they straddle lane boundaries.
        let tieRun = (threshold / 2)..<(threshold / 2 + 40)
        if let last = tieRun.last {
            for index in tieRun {
                vectors[index] = vectors[last]
            }
        }

        let buffer = try VectorBuffer(capacity: count, dim: dim, device: context.device)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(count)

        let queries = seededVectors(count: 3, dim: dim, seed: 99)
        for (queryIndex, query) in queries.enumerated() {
            let result = try await FlatGPUSearch.search(
                context: context, query: query, vectors: buffer, k: 24, metric: .cosine,
                tierOverride: 0)
            #expect(result.count == 24)
            let expected = bruteForceTopK(query: query, vectors: vectors, k: 24, metric: .cosine)
            #expect(
                Set(result.map(\.internalID)) == expected,
                "parallel single-query mismatch q=\(queryIndex)")
            #expect(
                result.map(\.score) == result.map(\.score).sorted(),
                "results not ascending q=\(queryIndex)")

            let batch = try await FlatGPUSearch.batchSearch(
                context: context, queries: [query], vectors: buffer, k: 24, metric: .cosine,
                tierOverride: 0)
            #expect(Set(batch[0].map(\.internalID)) == expected, "parallel batch mismatch q=\(queryIndex)")
        }
    }

    /// Concurrent host-tier searches share the corpus-norm cache; interleaved
    /// invalidation must never surface stale norms.
    @Test("Concurrent host searches with norm-cache churn stay exact")
    func concurrentHostSearchesSurviveCacheChurn() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        let count = 2_000
        let dim = 32
        let vectors = seededVectors(count: count, dim: dim, seed: 1_337)
        let buffer = try VectorBuffer(capacity: count, dim: dim, device: MTLCreateSystemDefaultDevice()!)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(count)

        let queries = seededVectors(count: 16, dim: dim, seed: 2_048)
        let expected: [[UInt32]] = queries.map {
            Array(bruteForceTopK(query: $0, vectors: vectors, k: 10, metric: .l2))
        }

        try await withThrowingTaskGroup(of: Bool.self) { group in
            for round in 0..<64 {
                group.addTask {
                    let results = try await FlatGPUSearch.search(
                        context: nil, query: queries[round % queries.count],
                        vectors: buffer, k: 10, metric: .l2,
                        tierOverride: Int.max)
                    return Set(results.prefix(10).map(\.internalID)) == Set(expected[round % queries.count])
                }
                if round % 7 == 0 {
                    group.addTask {
                        FlatGPUSearch.invalidateHostNormCache(buffer: buffer.buffer)
                        return true
                    }
                }
            }
            for try await ok in group {
                #expect(ok, "concurrent search diverged from brute force under cache churn")
            }
        }
    }

    // MARK: - Fallback behavior

    @Test("Disabling exact search falls back to graph traversal with usable recall")
    func disabledExactPathFallsBackToGraph() async throws {
        let config = IndexConfiguration(degree: 16, efSearch: 96, exactSearchMaxVectorCount: 0)
        let index = GraphIndex(configuration: config)
        let count = 800
        let dim = 64
        let vectors = seededVectors(count: count, dim: dim, seed: 91)
        let ids = (0..<count).map { "v_\($0)" }
        try await index.build(vectors: vectors, ids: ids)

        let queries = seededVectors(count: 6, dim: dim, seed: 303)
        var recallSum = 0.0
        for query in queries {
            let expected = bruteForceTopK(query: query, vectors: vectors, k: 10, metric: .cosine)
            let results = try await index.search(query: query, k: 10)
            let got = Set(results.prefix(10).map(\.internalID))
            recallSum += Double(got.intersection(expected).count) / 10.0
        }
        let meanRecall = recallSum / Double(queries.count)
        #expect(meanRecall >= 0.85, "graph fallback recall too low: \(meanRecall)")
    }

    @Test("Exact path disabled at runtime limit keeps results valid")
    func runtimeLimitFallbackKeepsResultsValid() async throws {
        let config = IndexConfiguration(degree: 16, efSearch: 96, exactSearchMaxVectorCount: 100)
        let index = GraphIndex(configuration: config)
        let count = 400
        let dim = 48
        let vectors = seededVectors(count: count, dim: dim, seed: 55)
        let ids = (0..<count).map { "v_\($0)" }
        try await index.build(vectors: vectors, ids: ids)

        let query = seededVectors(count: 1, dim: dim, seed: 65)[0]
        let results = try await index.search(query: query, k: 10)
        #expect(results.count == 10)
        #expect(results.allSatisfy { $0.internalID < UInt32(count) })
    }

    // MARK: - Updates / deletes / persistence through the public index API

    @Test("Recall stays exact across inserts, deletes, and reload")
    func exactnessAcrossLifecycle() async throws {
        let config = IndexConfiguration(degree: 16, metric: .cosine)
        let index = GraphIndex(configuration: config)
        let baseCount = 500
        let extraCount = 200
        let dim = 64
        let baseVectors = seededVectors(count: baseCount, dim: dim, seed: 12_345)
        let extraVectors = seededVectors(count: extraCount, dim: dim, seed: 54_321)
        let baseIDs = (0..<baseCount).map { "v_\($0)" }
        let extraIDs = (0..<extraCount).map { "new_\($0)" }

        try await index.build(vectors: baseVectors, ids: baseIDs)

        // Inserts preserve exactness including newly inserted rows.
        try await index.batchInsert(extraVectors, ids: extraIDs)
        let fullCorpus = baseVectors + extraVectors
        try await assertExactRecall(
            index: index, corpus: fullCorpus, queries: 5, dim: dim, seed: 7_001,
            stageLabel: "afterInsert")

        // Deletes remove rows from results and keep the rest exact.
        let deletedSlots = Set([3, 17, 240, 499])
        for slot in deletedSlots {
            try await index.delete(id: "v_\(slot)")
        }
        try await assertExactRecall(
            index: index, corpus: fullCorpus, queries: 5, dim: dim, seed: 7_002,
            excludedInternalIDs: deletedSlots, stageLabel: "afterDelete")
        for slot in deletedSlots {
            let probe = try await index.search(query: fullCorpus[slot], k: 20)
            #expect(!probe.contains { $0.id == "v_\(slot)" }, "deleted id v_\(slot) returned")
        }

        // Persistence round trip preserves exactness (mmap-compatible format).
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("exact-lifecycle-\(UUID().uuidString).annindex")
        defer { try? FileManager.default.removeItem(at: url) }
        try await index.saveMmapCompatible(to: url)

        let reloaded = try await GraphIndex.load(from: url)
        try await assertExactRecall(
            index: reloaded, corpus: fullCorpus, queries: 5, dim: dim, seed: 7_003,
            excludedInternalIDs: deletedSlots, stageLabel: "reloaded")

        let mmapped = try await GraphIndex.loadMmap(from: url)
        try await assertExactRecall(
            index: mmapped, corpus: fullCorpus, queries: 5, dim: dim, seed: 7_004,
            excludedInternalIDs: deletedSlots, stageLabel: "reloadedMmap")
    }

    private func assertExactRecall(
        index: GraphIndex,
        corpus: [[Float]],
        queries: Int,
        dim: Int,
        seed: UInt64,
        excludedInternalIDs: Set<Int> = [],
        stageLabel: String
    ) async throws {
        let probes = seededVectors(count: queries, dim: dim, seed: seed)
        for (queryIndex, query) in probes.enumerated() {
            let expected = bruteForceTopK(
                query: query, vectors: corpus, k: 10, metric: .cosine,
                skipIndices: excludedInternalIDs)
            let results = try await index.search(query: query, k: 10)
            let got = Set(results.prefix(10).map(\.internalID))
            #expect(
                got == expected,
                "[\(stageLabel)] exactness violated q=\(queryIndex): got \(got.sorted()), want \(expected.sorted())")
        }
    }

    private func makeContextGuarded() throws -> MetalContext {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        return try MetalContext()
    }
}
