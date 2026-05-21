import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Index Metrics Tests")
struct IndexMetricsTests {
    @Test("Metrics disabled by default")
    func metricsDisabledByDefault() async throws {
        let (index, vectors, ids) = try await makeIndexFixture(count: 12, dim: 8, seedOffset: 0)
        #expect(await index.metrics == nil)

        _ = try await index.search(query: vectors[0], k: 5)
        try await index.insert(makeVector(row: 10_000, dim: 8), id: "extra")
        _ = try await index.rangeSearch(query: vectors[1], maxDistance: 10_000, limit: 5)

        let streaming = Advanced.StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 5,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 8, metric: .cosine, hnswConfiguration: .init(enabled: false))
        ))
        #expect(await streaming.metrics == nil)

        try await streaming.batchInsert(vectors, ids: ids)
        try await streaming.flush()
        _ = try await streaming.search(query: vectors[2], k: 3)
    }

    @Test("Search count increments")
    func searchCountIncrements() async throws {
        let (index, vectors, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        for i in 0..<10 {
            _ = try await index.search(query: vectors[i], k: 5)
        }

        let snapshot = await metrics.snapshot()
        #expect(snapshot.searchCount == 10)
    }

    @Test("Range search increments search count")
    func rangeSearchCountIncrements() async throws {
        let (index, vectors, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        for i in 0..<7 {
            _ = try await index.rangeSearch(query: vectors[i], maxDistance: 10_000, limit: 8)
        }

        let snapshot = await metrics.snapshot()
        #expect(snapshot.searchCount == 7)
        let histogram = await metrics.searchLatencyHistogram
        #expect(histogram.reduce(0, +) == 7)
    }

    @Test("Insert count increments")
    func insertCountIncrements() async throws {
        let (index, _, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        for i in 0..<5 {
            try await index.insert(makeVector(row: 30_000 + i, dim: 16), id: "new-\(i)")
        }

        let snapshot = await metrics.snapshot()
        #expect(snapshot.insertCount == 5)
    }

    @Test("Batch insert counts vectors")
    func batchInsertCountsVectors() async throws {
        let (index, _, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        let newVectors = makeVectors(count: 20, dim: 16, seedOffset: 40_000)
        let newIDs = (0..<20).map { "batch-\($0)" }
        try await index.batchInsert(newVectors, ids: newIDs)

        let snapshot = await metrics.snapshot()
        #expect(snapshot.insertCount == 20)
    }

    @Test("Batch search count increments")
    func batchSearchCountIncrements() async throws {
        let (index, vectors, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        for pass in 0..<3 {
            let start = pass * 5
            let queries = Array(vectors[start..<(start + 5)])
            _ = try await index.batchSearch(queries: queries, k: 5)
        }

        let snapshot = await metrics.snapshot()
        #expect(snapshot.batchSearchCount == 3)
        #expect(snapshot.searchCount == 15)
    }

    @Test("Latency histogram populated")
    func latencyHistogramPopulated() async throws {
        let (index, vectors, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        for i in 0..<50 {
            let q = vectors[i % vectors.count]
            _ = try await index.search(query: q, k: 5)
        }

        let histogram = await metrics.searchLatencyHistogram
        #expect(histogram.reduce(0, +) == 50)
        #expect(histogram.contains(where: { $0 > 0 }))
    }

    @Test("Snapshot serializes to JSON")
    func snapshotSerializesToJSON() async throws {
        let (index, vectors, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)

        for i in 0..<5 {
            _ = try await index.search(query: vectors[i], k: 5)
        }
        try await index.insert(makeVector(row: 50_000, dim: 16), id: "ins-a")
        try await index.insert(makeVector(row: 50_001, dim: 16), id: "ins-b")

        let snapshot = await metrics.snapshot()
        let encoded = try JSONEncoder().encode(snapshot)
        let decoded = try JSONDecoder().decode(MetricsSnapshot.self, from: encoded)

        #expect(decoded.searchCount == 5)
        #expect(decoded.insertCount == 2)
        #expect(decoded.searchP50LatencyMs >= 0)
    }

    @Test("Streaming merge recorded")
    func streamingMergeRecorded() async throws {
        let streaming = Advanced.StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 5,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 8, metric: .cosine, hnswConfiguration: .init(enabled: false))
        ))
        let metrics = IndexMetrics()
        await streaming.setMetrics(metrics)

        for i in 0..<16 {
            try await streaming.insert(makeVector(row: 60_000 + i, dim: 16), id: "s-\(i)")
        }
        try await streaming.flush()

        let snapshot = await metrics.snapshot()
        #expect(snapshot.mergeCount >= 1)
    }

    @Test("Streaming single-record merge path not recorded")
    func streamingSingleRecordMergeNotRecorded() async throws {
        let streaming = Advanced.StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 5,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 8, metric: .cosine, hnswConfiguration: .init(enabled: false))
        ))
        let metrics = IndexMetrics()
        await streaming.setMetrics(metrics)

        try await streaming.insert(makeVector(row: 65_000, dim: 16), id: "single")
        try await streaming.flush()

        let snapshot = await metrics.snapshot()
        #expect(snapshot.mergeCount == 0)
    }

    @Test("Shared metrics across indexes")
    func sharedMetricsAcrossIndexes() async throws {
        let shared = IndexMetrics()

        let (index, vectors, _) = try await makeIndexFixture()
        await index.setMetrics(shared)

        let streaming = Advanced.StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 10,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 8, metric: .cosine, hnswConfiguration: .init(enabled: false))
        ))
        await streaming.setMetrics(shared)

        let streamVectors = makeVectors(count: 12, dim: 16, seedOffset: 70_000)
        let streamIDs = (0..<12).map { "st-\($0)" }
        try await streaming.batchInsert(streamVectors, ids: streamIDs)
        try await streaming.flush()

        for i in 0..<3 {
            _ = try await index.search(query: vectors[i], k: 5)
        }
        for i in 0..<2 {
            _ = try await streaming.search(query: streamVectors[i], k: 5)
        }

        let snapshot = await shared.snapshot()
        #expect(snapshot.searchCount == 5)
    }

    @Test("GraphIndex metrics not persisted")
    func anIndexMetricsNotPersistedAfterLoad() async throws {
        let (index, vectors, _) = try await makeIndexFixture()
        let metrics = IndexMetrics()
        await index.setMetrics(metrics)
        _ = try await index.search(query: vectors[0], k: 5)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-metrics-index-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let tempMetaURL = URL(fileURLWithPath: tempURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
        }

        try await index.save(to: tempURL)
        let loaded = try await GraphIndex.load(from: tempURL)

        #expect(await loaded.metrics == nil)
    }

    @Test("Streaming metrics not persisted")
    func streamingMetricsNotPersistedAfterLoad() async throws {
        let streaming = Advanced.StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 6,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 8, metric: .cosine, hnswConfiguration: .init(enabled: false))
        ))
        let metrics = IndexMetrics()
        await streaming.setMetrics(metrics)

        let vectors = makeVectors(count: 10, dim: 16, seedOffset: 80_000)
        let ids = (0..<10).map { "p-\($0)" }
        try await streaming.batchInsert(vectors, ids: ids)
        try await streaming.flush()

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-streaming-metrics-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        try await streaming.save(to: dir)
        let loaded = try await Advanced.StreamingIndex.load(from: dir)

        #expect(await loaded.metrics == nil)
    }

    private func makeIndexFixture(
        count: Int = 50,
        dim: Int = 16,
        seedOffset: Int = 0
    ) async throws -> (index: GraphIndex, vectors: [[Float]], ids: [String]) {
        let vectors = makeVectors(count: count, dim: dim, seedOffset: seedOffset)
        let ids = (0..<count).map { "v\($0)" }
        let index = GraphIndex(configuration: IndexConfiguration(
            degree: 8,
            metric: .cosine,
            hnswConfiguration: .init(enabled: false)
        ))
        try await index.build(vectors: vectors, ids: ids)
        return (index, vectors, ids)
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.113) + cos(i * 0.071)
            }
        }
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.097) + cos(i * 0.053)
        }
    }
}
