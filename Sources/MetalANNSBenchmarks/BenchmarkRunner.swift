import Foundation
import MetalANNS
import MetalANNSCore

struct BenchmarkRunner {
    struct Config {
        var vectorCount: Int = 1000
        var dim: Int = 128
        var degree: Int = 32
        var queryCount: Int = 100
        var k: Int = 10
        var efSearch: Int = 64
        var metric: Metric = .cosine
    }

    struct Results {
        var buildTimeMs: Double
        var queryLatencyP50Ms: Double
        var queryLatencyP95Ms: Double
        var queryLatencyP99Ms: Double
        var queryLatencyMeanMs: Double
        var queryLatencyStdDevMs: Double
        var queryLatencyMinMs: Double
        var queryLatencyMaxMs: Double
        var recallAt1: Double
        var recallAt10: Double
        var recallAt100: Double
        var queryCount: Int = 0
        var totalSearchTimeSeconds: Double = 0
        var latencyDistribution: LatencyDistribution = .empty
        var memoryBeforeBuild: MemorySnapshot = .zero()
        var memoryAfterBuild: MemorySnapshot = .zero()
        var memoryAfterQueries: MemorySnapshot = .zero()
        var firstQueryLatencyMs: Double = 0
        var warmSteadyMeanMs: Double = 0
        var concurrency: Int = 1

        var indexResidentBytesEstimate: UInt64 {
            memoryAfterBuild.residentBytes &- min(memoryAfterBuild.residentBytes, memoryBeforeBuild.residentBytes)
        }
    }

    static func run(config: Config, repeatRuns: Int = 1, warmupRuns: Int = 0) async throws -> Results {
        let normalizedConfig = normalize(config)
        let vectors = makeVectors(count: normalizedConfig.vectorCount, dim: normalizedConfig.dim, seedOffset: 0)
        let ids = (0..<normalizedConfig.vectorCount).map { "v_\($0)" }
        let queries = makeVectors(
            count: normalizedConfig.queryCount,
            dim: normalizedConfig.dim,
            seedOffset: 1_000_000
        )

        let targetNeighborCount = max(1, min(100, normalizedConfig.vectorCount))
        let expectedNeighbors = queries.map {
            bruteForceTopK(
                query: $0,
                vectors: vectors,
                k: targetNeighborCount,
                metric: normalizedConfig.metric
            )
        }

        return try await runIndexedBenchmark(
            config: normalizedConfig,
            indexVectors: vectors,
            indexIDs: ids,
            queries: queries,
            expectedNeighbors: expectedNeighbors,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns
        )
    }

    static func run(
        config: Config,
        dataset: BenchmarkDataset,
        repeatRuns: Int = 1,
        warmupRuns: Int = 0
    ) async throws -> Results {
        let normalizedConfig = normalize(config)

        guard normalizedConfig.queryCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("queryCount must be greater than zero")
        }
        guard !dataset.trainVectors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("trainVectors cannot be empty")
        }
        guard !dataset.testVectors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("testVectors cannot be empty")
        }

        let effectiveQueryCount = min(normalizedConfig.queryCount, dataset.testVectors.count)
        guard effectiveQueryCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("queryCount exceeds dataset query count")
        }

        let queries = Array(dataset.testVectors.prefix(effectiveQueryCount))
        let expectedNeighbors = Array(dataset.groundTruth.prefix(effectiveQueryCount))
        guard !expectedNeighbors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("groundTruth cannot be empty")
        }

        let ids = (0..<dataset.trainVectors.count).map { "v_\($0)" }

        return try await runIndexedBenchmark(
            config: normalizedConfig,
            indexVectors: dataset.trainVectors,
            indexIDs: ids,
            queries: queries,
            expectedNeighbors: expectedNeighbors,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns
        )
    }

    static func runConcurrent(
        config: Config,
        dataset: BenchmarkDataset,
        concurrency: Int,
        repeatRuns: Int = 1,
        warmupRuns: Int = 0
    ) async throws -> Results {
        let normalizedConfig = normalize(config)

        guard normalizedConfig.queryCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("queryCount must be greater than zero")
        }
        guard !dataset.trainVectors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("trainVectors cannot be empty")
        }
        guard !dataset.testVectors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("testVectors cannot be empty")
        }

        let effectiveQueryCount = min(normalizedConfig.queryCount, dataset.testVectors.count)
        guard effectiveQueryCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("queryCount exceeds dataset query count")
        }

        let queries = Array(dataset.testVectors.prefix(effectiveQueryCount))
        let expectedNeighbors = Array(dataset.groundTruth.prefix(effectiveQueryCount))
        guard !expectedNeighbors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("groundTruth cannot be empty")
        }

        let ids = (0..<dataset.trainVectors.count).map { "v_\($0)" }

        return try await runIndexedBenchmark(
            config: normalizedConfig,
            indexVectors: dataset.trainVectors,
            indexIDs: ids,
            queries: queries,
            expectedNeighbors: expectedNeighbors,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns,
            concurrency: max(1, concurrency)
        )
    }

    static func concurrencySweep(
        config: Config,
        dataset: BenchmarkDataset,
        concurrencyLevels: [Int],
        repeatRuns: Int = 1,
        warmupRuns: Int = 0
    ) async throws -> BenchmarkReport {
        var rows: [BenchmarkReport.Row] = []
        rows.reserveCapacity(concurrencyLevels.count)

        for level in concurrencyLevels {
            let effectiveLevel = max(1, level)
            let result = try await runConcurrent(
                config: config,
                dataset: dataset,
                concurrency: effectiveLevel,
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            )
            rows.append(
                BenchmarkReport.Row(
                    label: "concurrency=\(effectiveLevel)",
                    recallAt10: result.recallAt10,
                    qps: result.qps,
                    buildTimeMs: result.buildTimeMs,
                    p50Ms: result.queryLatencyP50Ms,
                    p95Ms: result.queryLatencyP95Ms,
                    p99Ms: result.queryLatencyP99Ms,
                    recallAt1: result.recallAt1,
                    recallAt100: result.recallAt100,
                    queryCount: result.queryCount,
                    avgQueryMs: result.queryLatencyMeanMs,
                    maxQueryMs: result.queryLatencyMaxMs,
                    p90Ms: result.latencyDistribution.p90Ms,
                    p999Ms: result.latencyDistribution.p999Ms,
                    stdDevMs: result.queryLatencyStdDevMs,
                    minMs: result.queryLatencyMinMs,
                    indexResidentMB: Double(result.indexResidentBytesEstimate) / (1024 * 1024),
                    peakResidentMB: result.memoryAfterQueries.peakResidentMB,
                    concurrency: effectiveLevel,
                    firstQueryMs: result.firstQueryLatencyMs,
                    warmSteadyMeanMs: result.warmSteadyMeanMs
                )
            )
        }

        return BenchmarkReport(
            rows: rows,
            datasetLabel: "train=\(dataset.trainVectors.count),test=\(dataset.testVectors.count),dim=\(dataset.dimension),metric=\(dataset.metric.rawValue)",
            metadata: [
                "queryCount": "\(config.queryCount)",
                "trainCount": "\(dataset.trainVectors.count)",
                "dimension": "\(dataset.dimension)",
                "metric": dataset.metric.rawValue,
                "concurrencySweep": concurrencyLevels.map(String.init).joined(separator: ",")
            ]
        )
    }

    /// Multi-backend comparator.
    ///
    /// IMPORTANT: `_GraphIndex` does not currently expose a public knob for
    /// selecting between its CPU / GPU / GPU-ADC search backends — the choice
    /// is made internally by `shouldUseHybridGPUSearch` based on workload size,
    /// metric, and EF parameters. As a result, this comparator runs the
    /// *same* index implementation for every requested label; the per-label
    /// rows it emits measure run-to-run variance of the auto-selected backend
    /// for that workload, not differences between distinct backends.
    /// `main.swift` prints a warning to that effect when `--compare` is
    /// invoked. The function is structured so that switching to per-label
    /// backend selection is a one-line change once such a public knob exists.
    static func compareBackends(
        config: Config,
        dataset: BenchmarkDataset,
        backendLabels: [String],
        repeatRuns: Int = 1,
        warmupRuns: Int = 0,
        concurrency: Int = 1
    ) async throws -> BenchmarkReport {
        var rows: [BenchmarkReport.Row] = []
        rows.reserveCapacity(backendLabels.count)

        for backendLabel in backendLabels {
            let result = try await runConcurrent(
                config: config,
                dataset: dataset,
                concurrency: max(1, concurrency),
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            )
            rows.append(
                BenchmarkReport.Row(
                    label: "backend=\(backendLabel)",
                    recallAt10: result.recallAt10,
                    qps: result.qps,
                    buildTimeMs: result.buildTimeMs,
                    p50Ms: result.queryLatencyP50Ms,
                    p95Ms: result.queryLatencyP95Ms,
                    p99Ms: result.queryLatencyP99Ms,
                    recallAt1: result.recallAt1,
                    recallAt100: result.recallAt100,
                    queryCount: result.queryCount,
                    avgQueryMs: result.queryLatencyMeanMs,
                    maxQueryMs: result.queryLatencyMaxMs,
                    p90Ms: result.latencyDistribution.p90Ms,
                    p999Ms: result.latencyDistribution.p999Ms,
                    stdDevMs: result.queryLatencyStdDevMs,
                    minMs: result.queryLatencyMinMs,
                    indexResidentMB: Double(result.indexResidentBytesEstimate) / (1024 * 1024),
                    peakResidentMB: result.memoryAfterQueries.peakResidentMB,
                    concurrency: max(1, concurrency),
                    firstQueryMs: result.firstQueryLatencyMs,
                    warmSteadyMeanMs: result.warmSteadyMeanMs,
                    backendLabel: backendLabel
                )
            )
        }

        return BenchmarkReport(
            rows: rows,
            datasetLabel: "train=\(dataset.trainVectors.count),test=\(dataset.testVectors.count),dim=\(dataset.dimension),metric=\(dataset.metric.rawValue)",
            metadata: [
                "queryCount": "\(config.queryCount)",
                "trainCount": "\(dataset.trainVectors.count)",
                "dimension": "\(dataset.dimension)",
                "metric": dataset.metric.rawValue,
                "compare": backendLabels.joined(separator: ","),
                "compareNote": "no public backend selector; rows reflect auto-selected backend variance"
            ]
        )
    }

    static func sweep(
        configs: [(label: String, config: Config)],
        dataset: BenchmarkDataset,
        repeatRuns: Int = 1,
        warmupRuns: Int = 0
    ) async throws -> BenchmarkReport {
        var rows: [BenchmarkReport.Row] = []
        rows.reserveCapacity(configs.count)

        for item in configs {
            let result = try await run(
                config: item.config,
                dataset: dataset,
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            )
            rows.append(
                BenchmarkReport.Row(
                    label: item.label,
                    recallAt10: result.recallAt10,
                    qps: result.qps,
                    buildTimeMs: result.buildTimeMs,
                    p50Ms: result.queryLatencyP50Ms,
                    p95Ms: result.queryLatencyP95Ms,
                    p99Ms: result.queryLatencyP99Ms,
                    recallAt1: result.recallAt1,
                    recallAt100: result.recallAt100,
                    queryCount: result.queryCount,
                    avgQueryMs: result.queryLatencyMeanMs,
                    maxQueryMs: result.queryLatencyMaxMs,
                    p90Ms: result.latencyDistribution.p90Ms,
                    p999Ms: result.latencyDistribution.p999Ms,
                    stdDevMs: result.queryLatencyStdDevMs,
                    minMs: result.queryLatencyMinMs,
                    indexResidentMB: Double(result.indexResidentBytesEstimate) / (1024 * 1024),
                    peakResidentMB: result.memoryAfterQueries.peakResidentMB
                )
            )
        }

        return BenchmarkReport(
            rows: rows,
            datasetLabel: "train=\(dataset.trainVectors.count),test=\(dataset.testVectors.count),dim=\(dataset.dimension),metric=\(dataset.metric.rawValue)",
            metadata: [
                "queryCount": "\(configs.first?.config.queryCount ?? 0)",
                "trainCount": "\(dataset.trainVectors.count)",
                "dimension": "\(dataset.dimension)",
                "metric": dataset.metric.rawValue,
                "sweep": "\(configs.count)"
            ]
        )
    }

    private static func runIndexedBenchmark(
        config: Config,
        indexVectors: [[Float]],
        indexIDs: [String],
        queries: [[Float]],
        expectedNeighbors: [[UInt32]],
        repeatRuns: Int,
        warmupRuns: Int,
        concurrency: Int = 1
    ) async throws -> Results {
        guard !indexVectors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("index vectors cannot be empty")
        }
        guard indexIDs.count == indexVectors.count else {
            throw BenchmarkDatasetError.invalidDataset("index vector and ID counts must match")
        }
        guard queries.count == expectedNeighbors.count else {
            throw BenchmarkDatasetError.invalidDataset("expectedNeighbors count must match query count")
        }
        guard !queries.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("queries cannot be empty")
        }

        let maxNeighborCount = expectedNeighbors.maxNeighborCount()
        guard maxNeighborCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("groundTruth rows must contain at least one entry")
        }

        let top1Count = min(1, maxNeighborCount)
        let top10Count = min(10, maxNeighborCount)
        let top100Count = min(100, maxNeighborCount)

        let index = _GraphIndex(
            configuration: IndexConfiguration(
                degree: config.degree,
                metric: config.metric,
                efSearch: config.efSearch
            )
        )

        let memoryBeforeBuild = MemorySnapshot.capture()
        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: indexVectors, ids: indexIDs)
        let buildEnd = DispatchTime.now().uptimeNanoseconds
        let buildTimeMs = Double(buildEnd - buildStart) / 1_000_000.0
        let memoryAfterBuild = MemorySnapshot.capture()

        let queryK = max(config.k, top100Count)
        let repeats = max(1, repeatRuns)
        let warmups = max(0, warmupRuns)
        let effectiveConcurrency = max(1, concurrency)

        let firstQuery = queries[0]
        let coldStart = DispatchTime.now().uptimeNanoseconds
        _ = try await index.search(query: firstQuery, k: queryK)
        let coldEnd = DispatchTime.now().uptimeNanoseconds
        let firstQueryLatencyMs = Double(coldEnd - coldStart) / 1_000_000.0

        if warmups > 0 {
            for _ in 0..<warmups {
                if effectiveConcurrency > 1 {
                    _ = try await benchmarkBatchConcurrent(
                        index: index,
                        queryK: queryK,
                        top1Count: top1Count,
                        top10Count: top10Count,
                        top100Count: top100Count,
                        queries: queries,
                        expectedNeighbors: expectedNeighbors,
                        concurrency: effectiveConcurrency,
                        measureLatency: false
                    )
                } else {
                    _ = try await benchmarkBatch(
                        index: index,
                        queryK: queryK,
                        top1Count: top1Count,
                        top10Count: top10Count,
                        top100Count: top100Count,
                        queries: queries,
                        expectedNeighbors: expectedNeighbors,
                        measureLatency: false
                    )
                }
            }
        }

        var allLatencies: [Double] = []
        allLatencies.reserveCapacity(queries.count * repeats)
        var recallAt1Total: Double = 0
        var recallAt10Total: Double = 0
        var recallAt100Total: Double = 0
        var totalSearchTimeSeconds = 0.0

        for _ in 0..<repeats {
            let batchStats: BenchmarkBatchStats
            if effectiveConcurrency > 1 {
                batchStats = try await benchmarkBatchConcurrent(
                    index: index,
                    queryK: queryK,
                    top1Count: top1Count,
                    top10Count: top10Count,
                    top100Count: top100Count,
                    queries: queries,
                    expectedNeighbors: expectedNeighbors,
                    concurrency: effectiveConcurrency,
                    measureLatency: true
                )
            } else {
                batchStats = try await benchmarkBatch(
                    index: index,
                    queryK: queryK,
                    top1Count: top1Count,
                    top10Count: top10Count,
                    top100Count: top100Count,
                    queries: queries,
                    expectedNeighbors: expectedNeighbors,
                    measureLatency: true
                )
            }

            recallAt1Total += batchStats.recallAt1
            recallAt10Total += batchStats.recallAt10
            recallAt100Total += batchStats.recallAt100

            allLatencies.append(contentsOf: batchStats.latencies)
            totalSearchTimeSeconds += batchStats.totalSearchSeconds
        }

        let memoryAfterQueries = MemorySnapshot.capture()
        let measuredQueryCount = Double(queries.count)
        let totalQueryCount = max(1, queries.count * repeats)
        let distribution = LatencyDistribution.compute(fromLatenciesMs: allLatencies)

        let warmSteadyMeanMs: Double
        if allLatencies.count > 1 {
            let warmSubset = allLatencies.dropFirst()
            warmSteadyMeanMs = warmSubset.reduce(0, +) / Double(warmSubset.count)
        } else {
            warmSteadyMeanMs = distribution.meanMs
        }

        return Results(
            buildTimeMs: buildTimeMs,
            queryLatencyP50Ms: distribution.p50Ms,
            queryLatencyP95Ms: distribution.p95Ms,
            queryLatencyP99Ms: distribution.p99Ms,
            queryLatencyMeanMs: distribution.meanMs,
            queryLatencyStdDevMs: distribution.stdDevMs,
            queryLatencyMinMs: distribution.minMs,
            queryLatencyMaxMs: distribution.maxMs,
            recallAt1: recallAt1Total / (measuredQueryCount * Double(repeats)),
            recallAt10: recallAt10Total / (measuredQueryCount * Double(repeats)),
            recallAt100: recallAt100Total / (measuredQueryCount * Double(repeats)),
            queryCount: totalQueryCount,
            totalSearchTimeSeconds: totalSearchTimeSeconds,
            latencyDistribution: distribution,
            memoryBeforeBuild: memoryBeforeBuild,
            memoryAfterBuild: memoryAfterBuild,
            memoryAfterQueries: memoryAfterQueries,
            firstQueryLatencyMs: firstQueryLatencyMs,
            warmSteadyMeanMs: warmSteadyMeanMs,
            concurrency: effectiveConcurrency
        )
    }

    private static func benchmarkBatch(
        index: _GraphIndex,
        queryK: Int,
        top1Count: Int,
        top10Count: Int,
        top100Count: Int,
        queries: [[Float]],
        expectedNeighbors: [[UInt32]],
        measureLatency: Bool
    ) async throws -> BenchmarkBatchStats {
        var latencies: [Double] = []
        if measureLatency {
            latencies.reserveCapacity(queries.count)
        }

        var recallAt1Total: Double = 0
        var recallAt10Total: Double = 0
        var recallAt100Total: Double = 0

        let batchStart = DispatchTime.now().uptimeNanoseconds
        for (queryIndex, query) in queries.enumerated() {
            let expected = expectedNeighbors[queryIndex]

            let searchStart = DispatchTime.now().uptimeNanoseconds
            let approx = try await index.search(query: query, k: queryK)
            let searchEnd = DispatchTime.now().uptimeNanoseconds

            if measureLatency {
                latencies.append(Double(searchEnd - searchStart) / 1_000_000.0)
            }

            let approxTop1 = BenchmarkIDParser.uint32Set(from: approx, limit: top1Count)
            let approxTop10 = BenchmarkIDParser.uint32Set(from: approx, limit: top10Count)
            let approxTop100 = BenchmarkIDParser.uint32Set(from: approx, limit: top100Count)

            let exactTop1 = Set(expected.prefix(top1Count))
            let exactTop10 = Set(expected.prefix(top10Count))
            let exactTop100 = Set(expected.prefix(top100Count))

            if !exactTop1.isEmpty {
                recallAt1Total += Double(approxTop1.intersection(exactTop1).count) / Double(exactTop1.count)
            }
            if !exactTop10.isEmpty {
                recallAt10Total += Double(approxTop10.intersection(exactTop10).count) / Double(exactTop10.count)
            }
            if !exactTop100.isEmpty {
                recallAt100Total += Double(approxTop100.intersection(exactTop100).count) / Double(exactTop100.count)
            }
        }
        let batchEnd = DispatchTime.now().uptimeNanoseconds

        return BenchmarkBatchStats(
            latencies: latencies,
            recallAt1: recallAt1Total,
            recallAt10: recallAt10Total,
            recallAt100: recallAt100Total,
            totalSearchSeconds: Double(batchEnd - batchStart) / 1_000_000_000.0
        )
    }

    private static func benchmarkBatchConcurrent(
        index: _GraphIndex,
        queryK: Int,
        top1Count: Int,
        top10Count: Int,
        top100Count: Int,
        queries: [[Float]],
        expectedNeighbors: [[UInt32]],
        concurrency: Int,
        measureLatency: Bool
    ) async throws -> BenchmarkBatchStats {
        let inflight = max(1, concurrency)
        let queryCount = queries.count
        let batchStart = DispatchTime.now().uptimeNanoseconds

        var collected: [PerQueryStats] = []
        collected.reserveCapacity(queryCount)

        try await withThrowingTaskGroup(of: PerQueryStats.self) { group in
            var dispatched = 0
            var completed = 0

            // Prime the sliding window with up to `inflight` in-flight tasks.
            let initialBurst = min(inflight, queryCount)
            for _ in 0..<initialBurst {
                let queryIndex = dispatched
                let query = queries[queryIndex]
                let expected = expectedNeighbors[queryIndex]
                group.addTask {
                    try await runOneQuery(
                        index: index,
                        query: query,
                        expected: expected,
                        queryK: queryK,
                        top1Count: top1Count,
                        top10Count: top10Count,
                        top100Count: top100Count,
                        measureLatency: measureLatency
                    )
                }
                dispatched += 1
            }

            while completed < queryCount {
                guard let stats = try await group.next() else {
                    break
                }
                collected.append(stats)
                completed += 1

                if dispatched < queryCount {
                    let queryIndex = dispatched
                    let query = queries[queryIndex]
                    let expected = expectedNeighbors[queryIndex]
                    group.addTask {
                        try await runOneQuery(
                            index: index,
                            query: query,
                            expected: expected,
                            queryK: queryK,
                            top1Count: top1Count,
                            top10Count: top10Count,
                            top100Count: top100Count,
                            measureLatency: measureLatency
                        )
                    }
                    dispatched += 1
                }
            }
        }

        let batchEnd = DispatchTime.now().uptimeNanoseconds

        var latencies: [Double] = []
        if measureLatency {
            latencies.reserveCapacity(collected.count)
        }
        var recallAt1Total: Double = 0
        var recallAt10Total: Double = 0
        var recallAt100Total: Double = 0
        for stats in collected {
            if measureLatency {
                latencies.append(stats.latencyMs)
            }
            recallAt1Total += stats.recallAt1
            recallAt10Total += stats.recallAt10
            recallAt100Total += stats.recallAt100
        }

        return BenchmarkBatchStats(
            latencies: latencies,
            recallAt1: recallAt1Total,
            recallAt10: recallAt10Total,
            recallAt100: recallAt100Total,
            totalSearchSeconds: Double(batchEnd - batchStart) / 1_000_000_000.0
        )
    }

    private static func runOneQuery(
        index: _GraphIndex,
        query: [Float],
        expected: [UInt32],
        queryK: Int,
        top1Count: Int,
        top10Count: Int,
        top100Count: Int,
        measureLatency: Bool
    ) async throws -> PerQueryStats {
        let searchStart = DispatchTime.now().uptimeNanoseconds
        let approx = try await index.search(query: query, k: queryK)
        let searchEnd = DispatchTime.now().uptimeNanoseconds

        let latencyMs = measureLatency
            ? Double(searchEnd - searchStart) / 1_000_000.0
            : 0

        let approxTop1 = BenchmarkIDParser.uint32Set(from: approx, limit: top1Count)
        let approxTop10 = BenchmarkIDParser.uint32Set(from: approx, limit: top10Count)
        let approxTop100 = BenchmarkIDParser.uint32Set(from: approx, limit: top100Count)

        let exactTop1 = Set(expected.prefix(top1Count))
        let exactTop10 = Set(expected.prefix(top10Count))
        let exactTop100 = Set(expected.prefix(top100Count))

        var r1: Double = 0
        var r10: Double = 0
        var r100: Double = 0
        if !exactTop1.isEmpty {
            r1 = Double(approxTop1.intersection(exactTop1).count) / Double(exactTop1.count)
        }
        if !exactTop10.isEmpty {
            r10 = Double(approxTop10.intersection(exactTop10).count) / Double(exactTop10.count)
        }
        if !exactTop100.isEmpty {
            r100 = Double(approxTop100.intersection(exactTop100).count) / Double(exactTop100.count)
        }

        return PerQueryStats(latencyMs: latencyMs, recallAt1: r1, recallAt10: r10, recallAt100: r100)
    }

    private static func normalize(_ config: Config) -> Config {
        var normalized = config
        normalized.queryCount = max(1, config.queryCount)
        normalized.vectorCount = max(1, config.vectorCount)
        normalized.dim = max(1, config.dim)
        normalized.degree = max(1, config.degree)
        normalized.k = max(1, config.k)
        normalized.efSearch = max(1, config.efSearch)
        return normalized
    }

    private static func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    private static func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> [UInt32] {
        let topCount = max(1, min(k, vectors.count))
        return vectors
            .enumerated()
            .map { idx, vector -> (id: UInt32, distance: Float) in
                (UInt32(idx), distance(query: query, vector: vector, metric: metric))
            }
            .sorted { $0.distance < $1.distance }
            .prefix(topCount)
            .map(\.id)
    }

    private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
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
            var mismatches = 0
            for d in 0..<query.count where query[d] != vector[d] {
                mismatches += 1
            }
            return Float(mismatches)
        }
    }

    private struct BenchmarkBatchStats {
        let latencies: [Double]
        let recallAt1: Double
        let recallAt10: Double
        let recallAt100: Double
        let totalSearchSeconds: Double
    }

    private struct PerQueryStats: Sendable {
        let latencyMs: Double
        let recallAt1: Double
        let recallAt10: Double
        let recallAt100: Double
    }
}

extension BenchmarkRunner.Results {
    var qps: Double {
        guard queryCount > 0, totalSearchTimeSeconds > 0 else {
            return 0
        }
        return Double(queryCount) / totalSearchTimeSeconds
    }
}

private extension Array where Element == [UInt32] {
    func maxNeighborCount() -> Int {
        reduce(into: 0) { count, row in
            count = Swift.max(count, row.count)
        }
    }
}
