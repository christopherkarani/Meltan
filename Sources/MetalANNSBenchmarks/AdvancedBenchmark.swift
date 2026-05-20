import Foundation
import MetalANNS
import MetalANNSCore

struct AdvancedBenchmark {
    static func storageSweep(
        config: BenchmarkRunner.Config,
        modes: [StorageBenchmarkMode],
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = syntheticDataset(config: config, seed: seed)
        let ids = ids(count: dataset.trainVectors.count)
        let effectiveK = max(config.k, min(100, dataset.trainVectors.count))
        let exact = exactNeighbors(
            queries: dataset.testVectors,
            vectors: dataset.trainVectors,
            allowedIDs: Set(ids),
            k: effectiveK,
            metric: config.metric
        )

        let index = _GraphIndex(configuration: indexConfiguration(from: config, count: dataset.trainVectors.count))
        let rssBeforeBuild = BenchmarkSystemMetrics.residentMemoryBytes()
        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: dataset.trainVectors, ids: ids)
        let buildMs = elapsedMs(since: buildStart)
        let rssAfterBuild = BenchmarkSystemMetrics.residentMemoryBytes()

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-storage-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let selectedModes = modes.isEmpty ? [.normal, .mmap, .diskBacked] : modes
        var rows: [BenchmarkReport.Row] = []

        if selectedModes.contains(.normal) || selectedModes.contains(.diskBacked) {
            let url = directory.appendingPathComponent("normal.anns")
            let saveStart = DispatchTime.now().uptimeNanoseconds
            try await index.save(to: url)
            let saveMs = elapsedMs(since: saveStart)
            let fileSize = graphIndexFileSize(at: url)

            if selectedModes.contains(.normal) {
                let loadRSS = BenchmarkSystemMetrics.residentMemoryBytes()
                let loadStart = DispatchTime.now().uptimeNanoseconds
                let loaded = try await _GraphIndex.load(from: url)
                let loadMs = elapsedMs(since: loadStart)
                let loadedRSS = BenchmarkSystemMetrics.residentMemoryBytes()
                rows.append(loadRow(
                    label: "storage=normal-load",
                    buildMs: buildMs,
                    operation: "load",
                    operationMs: loadMs,
                    requestedK: config.k,
                    effectiveK: effectiveK,
                    rssBefore: loadRSS,
                    rssAfter: loadedRSS,
                    fileSize: fileSize,
                    estimatedBackendPath: "normal-load"
                ))
                rows.append(try await firstQueryRow(
                    label: "storage=normal-first-query",
                    index: loaded,
                    dataset: dataset,
                    exact: exact,
                    buildMs: buildMs,
                    requestedK: config.k,
                    effectiveK: effectiveK,
                    fileSize: fileSize,
                    estimatedBackendPath: "normal-loaded"
                ))
                let warm = try await measureGraphSearches(
                    index: loaded,
                    queries: dataset.testVectors,
                    expected: exact,
                    k: effectiveK,
                    metric: config.metric,
                    repeatRuns: repeatRuns,
                    warmupRuns: warmupRuns
                )
                rows.append(row(
                    label: "storage=normal-warm-query",
                    stats: warm,
                    buildTimeMs: buildMs,
                    requestedK: config.k,
                    effectiveK: effectiveK,
                    operation: "warm-query",
                    estimatedBackendPath: "normal-loaded",
                    rssBefore: rssBeforeBuild,
                    rssAfter: BenchmarkSystemMetrics.residentMemoryBytes(),
                    fileSize: fileSize
                ))
            }

            if selectedModes.contains(.diskBacked) {
                let loadRSS = BenchmarkSystemMetrics.residentMemoryBytes()
                let loadStart = DispatchTime.now().uptimeNanoseconds
                let loaded = try await _GraphIndex.loadDiskBacked(from: url)
                let loadMs = elapsedMs(since: loadStart)
                let loadedRSS = BenchmarkSystemMetrics.residentMemoryBytes()
                rows.append(loadRow(
                    label: "storage=disk-backed-load",
                    buildMs: buildMs,
                    operation: "load",
                    operationMs: loadMs,
                    requestedK: config.k,
                    effectiveK: effectiveK,
                    rssBefore: loadRSS,
                    rssAfter: loadedRSS,
                    fileSize: fileSize,
                    estimatedBackendPath: "disk-backed"
                ))
                let warm = try await measureGraphSearches(
                    index: loaded,
                    queries: dataset.testVectors,
                    expected: exact,
                    k: effectiveK,
                    metric: config.metric,
                    repeatRuns: repeatRuns,
                    warmupRuns: warmupRuns
                )
                rows.append(row(
                    label: "storage=disk-backed-warm-query",
                    stats: warm,
                    buildTimeMs: buildMs,
                    requestedK: config.k,
                    effectiveK: effectiveK,
                    operation: "warm-query",
                    estimatedBackendPath: "disk-backed",
                    rssBefore: loadRSS,
                    rssAfter: BenchmarkSystemMetrics.residentMemoryBytes(),
                    fileSize: fileSize
                ))
            }

            rows.append(loadRow(
                label: "storage=normal-save",
                buildMs: buildMs,
                operation: "save",
                operationMs: saveMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                rssBefore: rssBeforeBuild,
                rssAfter: rssAfterBuild,
                fileSize: fileSize,
                estimatedBackendPath: "normal-save"
            ))
        }

        if selectedModes.contains(.mmap) {
            let url = directory.appendingPathComponent("mmap.mann")
            let saveStart = DispatchTime.now().uptimeNanoseconds
            try await index.saveMmapCompatible(to: url)
            let saveMs = elapsedMs(since: saveStart)
            let fileSize = graphIndexFileSize(at: url)

            rows.append(loadRow(
                label: "storage=mmap-save-v3",
                buildMs: buildMs,
                operation: "save",
                operationMs: saveMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                rssBefore: rssBeforeBuild,
                rssAfter: BenchmarkSystemMetrics.residentMemoryBytes(),
                fileSize: fileSize,
                estimatedBackendPath: "mmap-save"
            ))

            let loadRSS = BenchmarkSystemMetrics.residentMemoryBytes()
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let loaded = try await _GraphIndex.loadMmap(from: url)
            let loadMs = elapsedMs(since: loadStart)
            let loadedRSS = BenchmarkSystemMetrics.residentMemoryBytes()
            rows.append(loadRow(
                label: "storage=mmap-load",
                buildMs: buildMs,
                operation: "load",
                operationMs: loadMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                rssBefore: loadRSS,
                rssAfter: loadedRSS,
                fileSize: fileSize,
                estimatedBackendPath: "mmap"
            ))
            rows.append(try await firstQueryRow(
                label: "storage=mmap-first-query",
                index: loaded,
                dataset: dataset,
                exact: exact,
                buildMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                fileSize: fileSize,
                estimatedBackendPath: "mmap"
            ))
            let warm = try await measureGraphSearches(
                index: loaded,
                queries: dataset.testVectors,
                expected: exact,
                k: effectiveK,
                metric: config.metric,
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            )
            rows.append(row(
                label: "storage=mmap-warm-query",
                stats: warm,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "warm-query",
                estimatedBackendPath: "mmap",
                rssBefore: loadRSS,
                rssAfter: BenchmarkSystemMetrics.residentMemoryBytes(),
                fileSize: fileSize
            ))
        }

        return (dataset, rows)
    }

    static func streaming(
        config: BenchmarkRunner.Config,
        batchSize: Int,
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = syntheticDataset(config: config, seed: seed)
        let ids = ids(count: dataset.trainVectors.count)
        let effectiveK = max(config.k, min(100, dataset.trainVectors.count))
        let streamingConfig = StreamingConfiguration(
            deltaCapacity: max(2, batchSize),
            mergeStrategy: .blocking,
            indexConfiguration: indexConfiguration(from: config, count: dataset.trainVectors.count)
        )
        let index = Advanced.StreamingIndex(config: streamingConfig)

        let rssBefore = BenchmarkSystemMetrics.residentMemoryBytes()
        let insertStart = DispatchTime.now().uptimeNanoseconds
        let chunkSize = max(1, batchSize)
        var offset = 0
        while offset < dataset.trainVectors.count {
            let end = min(offset + chunkSize, dataset.trainVectors.count)
            try await index.batchInsert(Array(dataset.trainVectors[offset..<end]), ids: Array(ids[offset..<end]))
            offset = end
        }
        let insertMs = elapsedMs(since: insertStart)
        let rssAfterInsert = BenchmarkSystemMetrics.residentMemoryBytes()

        let flushStart = DispatchTime.now().uptimeNanoseconds
        try await index.flush()
        let flushMs = elapsedMs(since: flushStart)
        let rssAfterFlush = BenchmarkSystemMetrics.residentMemoryBytes()

        let exact = exactNeighbors(
            queries: dataset.testVectors,
            vectors: dataset.trainVectors,
            allowedIDs: Set(ids),
            k: effectiveK,
            metric: config.metric
        )
        let stats = try await measureSearches(
            queries: dataset.testVectors,
            expected: exact,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns
        ) { query in
            try await index.search(query: query, k: effectiveK, metric: config.metric)
        }

        return (dataset, [
            loadRow(
                label: "streaming=ingest",
                buildMs: 0,
                operation: "ingest",
                operationMs: insertMs,
                qps: insertMs > 0 ? Double(dataset.trainVectors.count) / (insertMs / 1000.0) : 0,
                queryCount: dataset.trainVectors.count,
                requestedK: config.k,
                effectiveK: effectiveK,
                rssBefore: rssBefore,
                rssAfter: rssAfterInsert,
                estimatedBackendPath: "streaming-delta"
            ),
            loadRow(
                label: "streaming=flush",
                buildMs: 0,
                operation: "flush",
                operationMs: flushMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                rssBefore: rssAfterInsert,
                rssAfter: rssAfterFlush,
                estimatedBackendPath: "streaming-merge"
            ),
            row(
                label: "streaming=warm-query",
                stats: stats,
                buildTimeMs: 0,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "warm-query",
                estimatedBackendPath: "streaming"
            )
        ])
    }

    static func sharded(
        config: BenchmarkRunner.Config,
        shards: Int,
        nprobe: Int,
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = syntheticDataset(config: config, seed: seed)
        let ids = ids(count: dataset.trainVectors.count)
        let effectiveK = max(config.k, min(100, dataset.trainVectors.count))
        let index = Advanced.ShardedIndex(
            numShards: shards,
            nprobe: nprobe,
            configuration: indexConfiguration(from: config, count: dataset.trainVectors.count)
        )

        let rssBefore = BenchmarkSystemMetrics.residentMemoryBytes()
        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: dataset.trainVectors, ids: ids)
        let buildMs = elapsedMs(since: buildStart)
        let rssAfter = BenchmarkSystemMetrics.residentMemoryBytes()

        let exact = exactNeighbors(
            queries: dataset.testVectors,
            vectors: dataset.trainVectors,
            allowedIDs: Set(ids),
            k: effectiveK,
            metric: config.metric
        )
        let stats = try await measureSearches(
            queries: dataset.testVectors,
            expected: exact,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns
        ) { query in
            try await index.search(query: query, k: effectiveK, metric: config.metric)
        }

        return (dataset, [
            loadRow(
                label: "sharded=build",
                buildMs: buildMs,
                operation: "build",
                operationMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                rssBefore: rssBefore,
                rssAfter: rssAfter,
                estimatedBackendPath: "sharded"
            ),
            row(
                label: "sharded=search",
                stats: stats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "search",
                estimatedBackendPath: "sharded"
            )
        ])
    }

    static func concurrent(
        config: BenchmarkRunner.Config,
        concurrency: Int,
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = syntheticDataset(config: config, seed: seed)
        let ids = ids(count: dataset.trainVectors.count)
        let effectiveK = max(config.k, min(100, dataset.trainVectors.count))
        let index = _GraphIndex(configuration: indexConfiguration(from: config, count: dataset.trainVectors.count))
        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: dataset.trainVectors, ids: ids)
        let buildMs = elapsedMs(since: buildStart)
        let exact = exactNeighbors(
            queries: dataset.testVectors,
            vectors: dataset.trainVectors,
            allowedIDs: Set(ids),
            k: effectiveK,
            metric: config.metric
        )

        if warmupRuns > 0 {
            _ = try await measureGraphSearches(
                index: index,
                queries: dataset.testVectors,
                expected: exact,
                k: effectiveK,
                metric: config.metric,
                repeatRuns: 1,
                warmupRuns: warmupRuns
            )
        }

        let stats = try await measureConcurrentSearches(
            index: index,
            queries: dataset.testVectors,
            expected: exact,
            k: effectiveK,
            metric: config.metric,
            repeatRuns: repeatRuns,
            concurrency: concurrency
        )

        return (dataset, [
            row(
                label: "concurrent=search",
                stats: stats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "concurrent-search",
                estimatedBackendPath: BenchmarkRunner.estimatedSearchBackendPath(
                    vectorCount: dataset.trainVectors.count,
                    degree: config.degree,
                    metric: config.metric,
                    k: effectiveK,
                    ef: max(config.efSearch, effectiveK)
                )
            )
        ])
    }

    static func unavailableComparatorRows(kind: String) -> [BenchmarkReport.Row] {
        [
            BenchmarkReport.Row(
                label: "\(kind)=unavailable",
                recallAt10: 0,
                qps: 0,
                buildTimeMs: 0,
                p50Ms: 0,
                p95Ms: 0,
                p99Ms: 0,
                operation: "\(kind)-unavailable",
                estimatedBackendPath: "unavailable"
            )
        ]
    }

    private static func firstQueryRow(
        label: String,
        index: _GraphIndex,
        dataset: BenchmarkDataset,
        exact: [[UInt32]],
        buildMs: Double,
        requestedK: Int,
        effectiveK: Int,
        fileSize: UInt64,
        estimatedBackendPath: String
    ) async throws -> BenchmarkReport.Row {
        let rssBefore = BenchmarkSystemMetrics.residentMemoryBytes()
        let start = DispatchTime.now().uptimeNanoseconds
        let results = try await index.search(query: dataset.testVectors[0], k: effectiveK, metric: dataset.metric)
        let firstMs = elapsedMs(since: start)
        let rssAfter = BenchmarkSystemMetrics.residentMemoryBytes()
        let stats = SearchStats(
            latencies: [firstMs],
            recallAt1: recall(results: results, expected: exact[0], count: min(1, exact[0].count)),
            recallAt10: recall(results: results, expected: exact[0], count: min(10, exact[0].count)),
            recallAt100: recall(results: results, expected: exact[0], count: min(100, exact[0].count)),
            totalSearchSeconds: firstMs / 1000.0,
            queryCount: 1
        )
        return row(
            label: label,
            stats: stats,
            buildTimeMs: buildMs,
            requestedK: requestedK,
            effectiveK: effectiveK,
            operation: "first-query",
            operationTimeMs: firstMs,
            estimatedBackendPath: estimatedBackendPath,
            rssBefore: rssBefore,
            rssAfter: rssAfter,
            fileSize: fileSize
        )
    }

    private static func measureGraphSearches(
        index: _GraphIndex,
        queries: [[Float]],
        expected: [[UInt32]],
        k: Int,
        metric: Metric,
        repeatRuns: Int,
        warmupRuns: Int
    ) async throws -> SearchStats {
        try await measureSearches(
            queries: queries,
            expected: expected,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns
        ) { query in
            try await index.search(query: query, k: k, metric: metric)
        }
    }

    private static func measureSearches(
        queries: [[Float]],
        expected: [[UInt32]],
        repeatRuns: Int,
        warmupRuns: Int,
        search: @escaping ([Float]) async throws -> [SearchResult]
    ) async throws -> SearchStats {
        let repeats = max(1, repeatRuns)
        let warmups = max(0, warmupRuns)
        if warmups > 0 {
            for _ in 0..<warmups {
                for query in queries {
                    _ = try await search(query)
                }
            }
        }

        var latencies: [Double] = []
        var recallAt1 = 0.0
        var recallAt10 = 0.0
        var recallAt100 = 0.0
        var totalSeconds = 0.0
        for _ in 0..<repeats {
            let batchStart = DispatchTime.now().uptimeNanoseconds
            for (index, query) in queries.enumerated() {
                let start = DispatchTime.now().uptimeNanoseconds
                let results = try await search(query)
                let elapsed = elapsedMs(since: start)
                latencies.append(elapsed)
                let row = expected[index]
                recallAt1 += recall(results: results, expected: row, count: min(1, row.count))
                recallAt10 += recall(results: results, expected: row, count: min(10, row.count))
                recallAt100 += recall(results: results, expected: row, count: min(100, row.count))
            }
            totalSeconds += Double(DispatchTime.now().uptimeNanoseconds - batchStart) / 1_000_000_000.0
        }

        let denominator = Double(max(1, queries.count * repeats))
        return SearchStats(
            latencies: latencies,
            recallAt1: recallAt1 / denominator,
            recallAt10: recallAt10 / denominator,
            recallAt100: recallAt100 / denominator,
            totalSearchSeconds: totalSeconds,
            queryCount: queries.count * repeats
        )
    }

    private static func measureConcurrentSearches(
        index: _GraphIndex,
        queries: [[Float]],
        expected: [[UInt32]],
        k: Int,
        metric: Metric,
        repeatRuns: Int,
        concurrency: Int
    ) async throws -> SearchStats {
        let repeats = max(1, repeatRuns)
        let width = max(1, concurrency)
        let indexedQueries = Array(repeating: Array(queries.enumerated()), count: repeats).flatMap { $0 }
        let batchStart = DispatchTime.now().uptimeNanoseconds
        var next = 0
        var stats: [ConcurrentSearchResult] = []
        stats.reserveCapacity(indexedQueries.count)

        try await withThrowingTaskGroup(of: ConcurrentSearchResult.self) { group in
            for _ in 0..<min(width, indexedQueries.count) {
                let item = indexedQueries[next]
                next += 1
                group.addTask {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let results = try await index.search(query: item.element, k: k, metric: metric)
                    return (item.offset, elapsedMs(since: start), results)
                }
            }

            for try await result in group {
                stats.append(result)
                if next < indexedQueries.count {
                    let item = indexedQueries[next]
                    next += 1
                    group.addTask {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let results = try await index.search(query: item.element, k: k, metric: metric)
                        return (item.offset, elapsedMs(since: start), results)
                    }
                }
            }
        }

        var recallAt1 = 0.0
        var recallAt10 = 0.0
        var recallAt100 = 0.0
        var latencies: [Double] = []
        latencies.reserveCapacity(stats.count)
        for (queryIndex, elapsed, results) in stats {
            latencies.append(elapsed)
            let row = expected[queryIndex]
            recallAt1 += recall(results: results, expected: row, count: min(1, row.count))
            recallAt10 += recall(results: results, expected: row, count: min(10, row.count))
            recallAt100 += recall(results: results, expected: row, count: min(100, row.count))
        }

        let denominator = Double(max(1, stats.count))
        return SearchStats(
            latencies: latencies,
            recallAt1: recallAt1 / denominator,
            recallAt10: recallAt10 / denominator,
            recallAt100: recallAt100 / denominator,
            totalSearchSeconds: Double(DispatchTime.now().uptimeNanoseconds - batchStart) / 1_000_000_000.0,
            queryCount: stats.count
        )
    }

    private static func loadRow(
        label: String,
        buildMs: Double,
        operation: String,
        operationMs: Double,
        qps: Double = 0,
        queryCount: Int = 0,
        requestedK: Int,
        effectiveK: Int,
        rssBefore: UInt64,
        rssAfter: UInt64,
        fileSize: UInt64 = 0,
        estimatedBackendPath: String
    ) -> BenchmarkReport.Row {
        BenchmarkReport.Row(
            label: label,
            recallAt10: 0,
            qps: qps,
            buildTimeMs: buildMs,
            p50Ms: 0,
            p95Ms: 0,
            p99Ms: 0,
            queryCount: queryCount,
            requestedK: requestedK,
            effectiveK: effectiveK,
            operation: operation,
            operationTimeMs: operationMs,
            estimatedBackendPath: estimatedBackendPath,
            rssBeforeBytes: rssBefore,
            rssAfterBytes: rssAfter,
            rssDeltaBytes: BenchmarkSystemMetrics.memoryDelta(before: rssBefore, after: rssAfter),
            fileSizeBytes: fileSize
        )
    }

    private static func row(
        label: String,
        stats: SearchStats,
        buildTimeMs: Double,
        requestedK: Int,
        effectiveK: Int,
        operation: String,
        operationTimeMs: Double = 0,
        estimatedBackendPath: String,
        rssBefore: UInt64 = 0,
        rssAfter: UInt64 = 0,
        fileSize: UInt64 = 0
    ) -> BenchmarkReport.Row {
        let sorted = stats.latencies.sorted()
        return BenchmarkReport.Row(
            label: label,
            recallAt10: stats.recallAt10,
            qps: stats.totalSearchSeconds > 0 ? Double(stats.queryCount) / stats.totalSearchSeconds : 0,
            buildTimeMs: buildTimeMs,
            p50Ms: percentile(0.50, in: sorted),
            p95Ms: percentile(0.95, in: sorted),
            p99Ms: percentile(0.99, in: sorted),
            recallAt1: stats.recallAt1,
            recallAt100: stats.recallAt100,
            queryCount: stats.queryCount,
            avgQueryMs: mean(in: sorted),
            maxQueryMs: sorted.last ?? 0,
            requestedK: requestedK,
            effectiveK: effectiveK,
            operation: operation,
            operationTimeMs: operationTimeMs,
            estimatedBackendPath: estimatedBackendPath,
            rssBeforeBytes: rssBefore,
            rssAfterBytes: rssAfter,
            rssDeltaBytes: BenchmarkSystemMetrics.memoryDelta(before: rssBefore, after: rssAfter),
            fileSizeBytes: fileSize
        )
    }

    private static func syntheticDataset(config: BenchmarkRunner.Config, seed: Int) -> BenchmarkDataset {
        BenchmarkDataset.synthetic(
            trainCount: max(2, config.vectorCount),
            testCount: max(1, config.queryCount),
            dimension: max(1, config.dim),
            k: min(100, max(2, config.vectorCount)),
            metric: config.metric,
            seed: seed
        )
    }

    private static func indexConfiguration(from config: BenchmarkRunner.Config, count: Int) -> IndexConfiguration {
        IndexConfiguration(
            degree: min(config.degree, max(1, count - 1)),
            metric: config.metric,
            efSearch: config.efSearch
        )
    }

    private static func exactNeighbors(
        queries: [[Float]],
        vectors: [[Float]],
        allowedIDs: Set<String>,
        k: Int,
        metric: Metric
    ) -> [[UInt32]] {
        let topCount = max(1, min(k, allowedIDs.count))
        return queries.map { query in
            vectors.enumerated()
                .compactMap { index, vector -> (id: UInt32, distance: Float)? in
                    let externalID = "v_\(index)"
                    guard allowedIDs.contains(externalID) else {
                        return nil
                    }
                    return (UInt32(index), SIMDDistance.distance(query, vector, metric: metric))
                }
                .sorted { $0.distance < $1.distance }
                .prefix(topCount)
                .map(\.id)
        }
    }

    private static func recall(results: [SearchResult], expected: [UInt32], count: Int) -> Double {
        guard count > 0 else {
            return 0
        }
        let actual = BenchmarkIDParser.uint32Set(from: results, limit: count)
        let exact = Set(expected.prefix(count))
        return Double(actual.intersection(exact).count) / Double(exact.count)
    }

    private static func ids(count: Int) -> [String] {
        (0..<count).map { "v_\($0)" }
    }

    private static func graphIndexFileSize(at url: URL) -> UInt64 {
        let databaseURL = url.deletingPathExtension().appendingPathExtension("db")
        let legacyDBURL = URL(fileURLWithPath: url.path + ".meta.db")
        return [
            url,
            databaseURL,
            URL(fileURLWithPath: databaseURL.path + "-wal"),
            URL(fileURLWithPath: databaseURL.path + "-shm"),
            URL(fileURLWithPath: url.path + ".meta.json"),
            legacyDBURL,
            URL(fileURLWithPath: legacyDBURL.path + "-wal"),
            URL(fileURLWithPath: legacyDBURL.path + "-shm"),
        ].reduce(UInt64(0)) { total, artifact in
            total + BenchmarkSystemMetrics.fileSizeBytes(at: artifact)
        }
    }

    private static func percentile(_ p: Double, in values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        let rank = Int(ceil(p * Double(values.count))) - 1
        return values[min(max(rank, 0), values.count - 1)]
    }

    private static func mean(in values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func elapsedMs(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
    }

    private struct SearchStats {
        let latencies: [Double]
        let recallAt1: Double
        let recallAt10: Double
        let recallAt100: Double
        let totalSearchSeconds: Double
        let queryCount: Int
    }

    private typealias ConcurrentSearchResult = (Int, Double, [SearchResult])
}
