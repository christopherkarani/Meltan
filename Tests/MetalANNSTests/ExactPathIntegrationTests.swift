import Foundation
import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

/// Public-API coverage for the fused exact-search path: recall against brute
/// force on both host and GPU tiers, plus fallback and mutation behavior
/// (inserts, deletes, compaction, persistence) at the `GraphIndex` level.
@Suite("Exact Path Integration Tests")
struct ExactPathIntegrationTests {
    // MARK: - Independent brute-force ground truth

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
            return denom < 1e-10 ? 1.0 : (1.0 - dot / denom)
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
            var mismatches = 0
            for d in 0..<query.count where query[d] != vector[d] {
                mismatches += 1
            }
            return Float(mismatches)
        }
    }

    private func bruteForceTopKIDs(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> Set<UInt32> {
        Set(
            vectors.enumerated()
                .map { (UInt32($0.offset), distance(query: query, vector: $0.element, metric: metric)) }
                .sorted { $0.1 < $1.1 }
                .prefix(k)
                .map(\.0)
        )
    }

    private func recall(
        _ results: [SearchResult],
        against expected: Set<UInt32>
    ) -> Double {
        guard !expected.isEmpty else { return 0 }
        let got = Set(results.map(\.internalID))
        return Double(got.intersection(expected).count) / Double(expected.count)
    }

    // MARK: - Host-tier exact recall (public API)

    @Test(
        "Exact path holds recall@k = 1.0 across metrics (host tier)",
        arguments: [MetalANNSCore.Metric.cosine, .l2, .innerProduct])
    func exactRecallHostTier(metric: MetalANNSCore.Metric) async throws {
        let n = 2_000
        let dim = 64
        let vectors = seededVectors(count: n, dim: dim, seed: UInt64(100 + n))
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 12, dim: dim, seed: 7_777)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: metric, efSearch: 64, maxIterations: 10)
        )
        try await index.build(vectors: vectors, ids: ids)

        for (queryIndex, query) in queries.enumerated() {
            let k = queryIndex % 2 == 0 ? 10 : 100
            let results = try await index.search(query: query, k: k)
            let expected = bruteForceTopKIDs(query: query, vectors: vectors, k: k, metric: metric)
            #expect(results.count == k)
            #expect(
                recall(results, against: expected) == 1.0,
                "exact path recall < 1.0 for metric \(metric.rawValue) query \(queryIndex)"
            )
            for index in 1..<results.count {
                #expect(results[index].score >= results[index - 1].score - 1e-6)
            }
        }
    }

    // MARK: - GPU-tier exact recall

    @Test("FlatGPUSearch GPU tier (above host cutoff) stays exact across metrics")
    func flatGPUSearchGPUTierExact() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        let context = try MetalContext()
        let n = 33_000  // Above hostTierMaxVectorCount (32_768) → GPU kernel path.
        let dim = 32
        let vectors = seededVectors(count: n, dim: dim, seed: 33_001)
        let buffer = try VectorBuffer(capacity: n, dim: dim, device: context.device)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(n)

        let queries = seededVectors(count: 4, dim: dim, seed: 33_002)
        for (queryIndex, query) in queries.enumerated() {
            let metric = [Metric.cosine, .l2, .innerProduct][queryIndex % 3]
            let results = try await FlatGPUSearch.search(
                context: context, query: query, vectors: buffer, k: 10, metric: metric
            )
            let expected = bruteForceTopKIDs(query: query, vectors: vectors, k: 10, metric: metric)
            #expect(results.count == 10)
            #expect(recall(results, against: expected) == 1.0, "GPU tier mismatch for \(metric.rawValue)")
        }
    }

    @Test("GraphIndex exact path stays exact on GPU-tier corpus sizes")
    func graphIndexGPUTierExact() async throws {
        let n = 33_000
        let dim = 32
        let vectors = seededVectors(count: n, dim: dim, seed: 33_100)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 3, dim: dim, seed: 33_101)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .cosine, efSearch: 64, maxIterations: 8)
        )
        try await index.build(vectors: vectors, ids: ids)

        for query in queries {
            let results = try await index.search(query: query, k: 10)
            let expected = bruteForceTopKIDs(query: query, vectors: vectors, k: 10, metric: .cosine)
            #expect(recall(results, against: expected) == 1.0)
        }
    }

    // MARK: - Fallback behavior

    @Test("Disabling the exact path falls back to graph search and stays correct")
    func fallbackWhenExactSearchDisabled() async throws {
        let n = 1_500
        let dim = 48
        let vectors = seededVectors(count: n, dim: dim, seed: 55_001)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 8, dim: dim, seed: 55_002)

        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 16, metric: .cosine, efSearch: 128, maxIterations: 10,
                exactSearchMaxVectorCount: 0  // Force the legacy graph path.
            )
        )
        try await index.build(vectors: vectors, ids: ids)

        for query in queries {
            let results = try await index.search(query: query, k: 10)
            #expect(results.count == 10)
            for index in 1..<results.count {
                #expect(results[index].score >= results[index - 1].score - 1e-6)
            }
            let expected = bruteForceTopKIDs(query: query, vectors: vectors, k: 10, metric: .cosine)
            #expect(recall(results, against: expected) == 1.0)
        }
    }

    @Test("Float16 storage is ineligible for the exact path and still answers")
    func float16StorageFallsBack() async throws {
        let n = 1_000
        let dim = 32
        let vectors = seededVectors(count: n, dim: dim, seed: 55_101)
        let ids = (0..<n).map { "v_\($0)" }
        let query = seededVector(dim: dim, seed: 55_102)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .cosine, efSearch: 64, useFloat16: true)
        )
        try await index.build(vectors: vectors, ids: ids)
        let results = try await index.search(query: query, k: 10)
        #expect(results.count == 10)
    }

    // MARK: - Mutation behavior

    @Test("Inserts after build keep the exact path exact (norm-cache invalidation)")
    func insertsStayExact() async throws {
        let n = 1_200
        let dim = 48
        var vectors = seededVectors(count: n, dim: dim, seed: 66_001)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 6, dim: dim, seed: 66_002)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .cosine, efSearch: 64, maxIterations: 10)
        )
        try await index.build(vectors: vectors, ids: ids)

        // Prime the exact path (and its host norm cache) before mutating.
        for query in queries {
            _ = try await index.search(query: query, k: 10)
        }

        let appended = seededVectors(count: 200, dim: dim, seed: 66_003)
        for (offset, vector) in appended.enumerated() {
            try await index.insert(vector, id: "u_\(offset)")
        }
        vectors.append(contentsOf: appended)

        for query in queries {
            let results = try await index.search(query: query, k: 10)
            // Appended rows land at internal IDs n..<n+200 and the original
            // rows keep 0..<n, so row index == internal ID throughout.
            let expected = bruteForceTopKIDs(query: query, vectors: vectors, k: 10, metric: .cosine)
            #expect(
                recall(results, against: expected) == 1.0,
                "stale results after append"
            )
        }
    }

    @Test("Deletes are filtered from results and the exact path stays exact")
    func deletesFilterAndStayCorrect() async throws {
        let n = 1_500
        let dim = 48
        let vectors = seededVectors(count: n, dim: dim, seed: 77_001)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 6, dim: dim, seed: 77_002)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .cosine, efSearch: 96, maxIterations: 10)
        )
        try await index.build(vectors: vectors, ids: ids)

        // Delete rows that actually appear in some query's top-10 so the
        // filter is exercised, plus a fixed spread of IDs. k + deletedCount
        // stays below the exact path's maxTopK (256), so the delete-aware
        // over-fetch keeps results exact.
        var deletedRows = Set<UInt32>()
        for query in queries {
            deletedRows.formUnion(
                Array(bruteForceTopKIDs(query: query, vectors: vectors, k: 10, metric: .cosine).prefix(3))
            )
        }
        for row in stride(from: 0, to: n, by: 97) {
            deletedRows.insert(UInt32(row))
        }
        for row in deletedRows {
            try await index.delete(id: "v_\(row)")
        }

        for query in queries {
            let results = try await index.search(query: query, k: 10)
            #expect(results.count == 10)
            let returnedRows = Set(results.map(\.internalID))
            #expect(
                returnedRows.isDisjoint(with: deletedRows),
                "deleted rows leaked into results"
            )
            // Ground truth: the 10 nearest non-deleted rows.
            let expected = Set(
                vectors.enumerated()
                    .filter { !deletedRows.contains(UInt32($0.offset)) }
                    .map { (UInt32($0.offset), distance(query: query, vector: $0.element, metric: .cosine)) }
                    .sorted { $0.1 < $1.1 }
                    .prefix(10)
                    .map(\.0)
            )
            #expect(recall(results, against: expected) == 1.0)
        }
    }

    @Test("Compact after deletes preserves exact recall on the exact path")
    func compactRestoresExactness() async throws {
        let n = 1_000
        let dim = 32
        let vectors = seededVectors(count: n, dim: dim, seed: 88_001)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 4, dim: dim, seed: 88_002)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .l2, efSearch: 64, maxIterations: 10)
        )
        try await index.build(vectors: vectors, ids: ids)

        for row in stride(from: 1, to: n, by: 3) {
            try await index.delete(id: "v_\(row)")
        }
        try await index.compact()

        for query in queries {
            let results = try await index.search(query: query, k: 10)
            // Compaction reassigns internal IDs; compare via external keys.
            let got = Set(results.map(\.id))
            let ranked = vectors.enumerated()
                .filter { $0.offset % 3 != 1 }
                .map { (UInt32($0.offset), distance(query: query, vector: $0.element, metric: .l2)) }
                .sorted { $0.1 < $1.1 }
                .prefix(10)
                .map { "v_\($0.0)" }
            #expect(got == Set(ranked))
        }
    }

    @Test("batchSearch filters deletes and stays exact via over-fetch")
    func batchSearchFiltersDeletes() async throws {
        let n = 1_200
        let dim = 40
        let vectors = seededVectors(count: n, dim: dim, seed: 78_001)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 8, dim: dim, seed: 78_002)

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .l2, efSearch: 64, maxIterations: 10)
        )
        try await index.build(vectors: vectors, ids: ids)

        var deletedRows = Set<UInt32>()
        for row in stride(from: 5, to: n, by: 61) {
            deletedRows.insert(UInt32(row))
        }
        for row in deletedRows {
            try await index.delete(id: "v_\(row)")
        }

        let batches = try await index.batchSearch(queries: queries, k: 10)
        #expect(batches.count == queries.count)
        for (queryIndex, batch) in batches.enumerated() {
            #expect(batch.count == 10)
            let returnedRows = Set(batch.map(\.internalID))
            #expect(returnedRows.isDisjoint(with: deletedRows), "deleted rows leaked into batch results")
            let expected = Set(
                vectors.enumerated()
                    .filter { !deletedRows.contains(UInt32($0.offset)) }
                    .map { (UInt32($0.offset), distance(query: queries[queryIndex], vector: $0.element, metric: .l2)) }
                    .sorted { $0.1 < $1.1 }
                    .prefix(10)
                    .map(\.0)
            )
            #expect(recall(batch, against: expected) == 1.0)
        }
    }

    // MARK: - Persistence

    @Test("Persistence round-trip preserves exact search results and configuration")
    func persistenceRoundTripPreservesExactSearch() async throws {
        let n = 1_000
        let dim = 40
        let vectors = seededVectors(count: n, dim: dim, seed: 99_001)
        let ids = (0..<n).map { "v_\($0)" }
        let queries = seededVectors(count: 5, dim: dim, seed: 99_002)

        let limit = 500_000
        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 16, metric: .cosine, efSearch: 64, maxIterations: 10,
                exactSearchMaxVectorCount: limit
            )
        )
        try await index.build(vectors: vectors, ids: ids)

        var before: [[SearchResult]] = []
        for query in queries {
            before.append(try await index.search(query: query, k: 10))
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-exact-path-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await index.save(to: tempURL)
        let loaded = try await GraphIndex.load(from: tempURL)

        #expect(await loaded.configuration.exactSearchMaxVectorCount == limit)

        for (queryIndex, query) in queries.enumerated() {
            let after = try await loaded.search(query: query, k: 10)
            #expect(after.map(\.id) == before[queryIndex].map(\.id))
            #expect(
                after.map(\.score).enumerated().allSatisfy { offset, score in
                    abs(score - before[queryIndex][offset].score) < 1e-6
                })
            let expected = bruteForceTopKIDs(query: query, vectors: vectors, k: 10, metric: .cosine)
            #expect(recall(after, against: expected) == 1.0)
        }
    }
}
