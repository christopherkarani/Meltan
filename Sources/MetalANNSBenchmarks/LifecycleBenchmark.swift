import Foundation
import MetalANNS
import MetalANNSCore

struct LifecycleBenchmark {
    static func filterSweep(
        config: BenchmarkRunner.Config,
        selectivities: [Double],
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: config.vectorCount,
            testCount: config.queryCount,
            dimension: config.dim,
            k: min(100, config.vectorCount),
            metric: config.metric,
            seed: seed
        )

        let index = _GraphIndex(configuration: IndexConfiguration(
            degree: min(config.degree, max(1, dataset.trainVectors.count - 1)),
            metric: config.metric,
            efSearch: config.efSearch
        ))
        let ids = ids(count: dataset.trainVectors.count)

        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: dataset.trainVectors, ids: ids)
        let buildMs = elapsedMs(since: buildStart)

        var rows: [BenchmarkReport.Row] = []
        rows.reserveCapacity(selectivities.count)

        for selectivity in selectivities {
            let matching = matchingIDs(total: dataset.trainVectors.count, ratio: selectivity)
            let matchingSet = Set(matching)
            let column = "bench_filter_\(ratioLabel(selectivity))"
            for id in matching {
                try await index.setMetadata(column, value: "match", for: id)
            }

            let exact = exactNeighbors(
                queries: dataset.testVectors,
                vectors: dataset.trainVectors,
                allowedIDs: matchingSet,
                k: min(100, matching.count),
                metric: config.metric
            )
            let effectiveK = max(config.k, exact.maxNeighborCount())
            let stats = try await measureSearches(
                queries: dataset.testVectors,
                expected: exact,
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            ) { query in
                try await index.search(
                    query: query,
                    k: effectiveK,
                    filter: .equals(column: column, value: "match"),
                    metric: config.metric
                )
            }
            let backend = BenchmarkRunner.estimatedSearchBackendPath(
                vectorCount: dataset.trainVectors.count,
                degree: config.degree,
                metric: config.metric,
                k: effectiveK,
                ef: max(config.efSearch, effectiveK)
            )

            rows.append(row(
                label: "filter=\(ratioLabel(selectivity))",
                stats: stats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "filter",
                estimatedBackendPath: "filtered-\(backend)"
            ))
        }

        return (dataset, rows)
    }

    static func deleteSweep(
        config: BenchmarkRunner.Config,
        deleteRatios: [Double],
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: config.vectorCount,
            testCount: config.queryCount,
            dimension: config.dim,
            k: min(100, config.vectorCount),
            metric: config.metric,
            seed: seed
        )
        let ids = ids(count: dataset.trainVectors.count)
        var rows: [BenchmarkReport.Row] = []
        rows.reserveCapacity(deleteRatios.count * 2)

        for ratio in deleteRatios {
            let index = _GraphIndex(configuration: IndexConfiguration(
                degree: min(config.degree, max(1, dataset.trainVectors.count - 1)),
                metric: config.metric,
                efSearch: config.efSearch
            ))

            let buildStart = DispatchTime.now().uptimeNanoseconds
            try await index.build(vectors: dataset.trainVectors, ids: ids)
            let buildMs = elapsedMs(since: buildStart)

            let deleted = deletionIDs(total: dataset.trainVectors.count, ratio: ratio)
            let deletedSet = Set(deleted)
            let deleteStart = DispatchTime.now().uptimeNanoseconds
            for id in deleted {
                try await index.delete(id: id)
            }
            let deleteMs = elapsedMs(since: deleteStart)

            let allowed = Set(ids).subtracting(deletedSet)
            let exact = exactNeighbors(
                queries: dataset.testVectors,
                vectors: dataset.trainVectors,
                allowedIDs: allowed,
                k: min(100, allowed.count),
                metric: config.metric
            )
            let effectiveK = max(config.k, exact.maxNeighborCount())
            let preStats = try await measureSearches(
                queries: dataset.testVectors,
                expected: exact,
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            ) { query in
                try await index.search(query: query, k: effectiveK, metric: config.metric)
            }

            rows.append(row(
                label: "delete=\(ratioLabel(ratio))-precompact",
                stats: preStats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "delete",
                operationTimeMs: deleteMs,
                estimatedBackendPath: "soft-delete"
            ))

            let compactStart = DispatchTime.now().uptimeNanoseconds
            try await index.compact()
            let compactMs = elapsedMs(since: compactStart)
            let postStats = try await measureSearches(
                queries: dataset.testVectors,
                expected: exact,
                repeatRuns: repeatRuns,
                warmupRuns: warmupRuns
            ) { query in
                try await index.search(query: query, k: effectiveK, metric: config.metric)
            }

            rows.append(row(
                label: "delete=\(ratioLabel(ratio))-postcompact",
                stats: postStats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "compact",
                operationTimeMs: compactMs,
                estimatedBackendPath: "compacted"
            ))
        }

        return (dataset, rows)
    }

    static func persistence(
        config: BenchmarkRunner.Config,
        repeatRuns: Int,
        warmupRuns: Int,
        seed: Int
    ) async throws -> (dataset: BenchmarkDataset, rows: [BenchmarkReport.Row]) {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: config.vectorCount,
            testCount: config.queryCount,
            dimension: config.dim,
            k: min(100, config.vectorCount),
            metric: config.metric,
            seed: seed
        )

        let index = _GraphIndex(configuration: IndexConfiguration(
            degree: min(config.degree, max(1, dataset.trainVectors.count - 1)),
            metric: config.metric,
            efSearch: config.efSearch
        ))
        let ids = ids(count: dataset.trainVectors.count)

        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: dataset.trainVectors, ids: ids)
        let buildMs = elapsedMs(since: buildStart)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-persistence-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("index.anns")

        let saveStart = DispatchTime.now().uptimeNanoseconds
        try await index.save(to: url)
        let saveMs = elapsedMs(since: saveStart)

        let loadStart = DispatchTime.now().uptimeNanoseconds
        let loaded = try await _GraphIndex.load(from: url)
        let loadMs = elapsedMs(since: loadStart)

        let exact = exactNeighbors(
            queries: dataset.testVectors,
            vectors: dataset.trainVectors,
            allowedIDs: Set(ids),
            k: min(100, dataset.trainVectors.count),
            metric: config.metric
        )
        let effectiveK = max(config.k, exact.maxNeighborCount())

        let firstQueryStart = DispatchTime.now().uptimeNanoseconds
        let firstResult = try await loaded.search(
            query: dataset.testVectors[0],
            k: effectiveK,
            metric: config.metric
        )
        let firstQueryMs = elapsedMs(since: firstQueryStart)
        let firstStats = SearchStats(
            latencies: [firstQueryMs],
            recallAt1: recall(results: firstResult, expected: exact[0], count: min(1, exact[0].count)),
            recallAt10: recall(results: firstResult, expected: exact[0], count: min(10, exact[0].count)),
            recallAt100: recall(results: firstResult, expected: exact[0], count: min(100, exact[0].count)),
            totalSearchSeconds: firstQueryMs / 1000.0,
            queryCount: 1
        )

        let warmStats = try await measureSearches(
            queries: dataset.testVectors,
            expected: exact,
            repeatRuns: repeatRuns,
            warmupRuns: warmupRuns
        ) { query in
            try await loaded.search(query: query, k: effectiveK, metric: config.metric)
        }

        return (dataset, [
            BenchmarkReport.Row(
                label: "persistence=save",
                recallAt10: 0,
                qps: 0,
                buildTimeMs: buildMs,
                p50Ms: 0,
                p95Ms: 0,
                p99Ms: 0,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "save",
                operationTimeMs: saveMs,
                estimatedBackendPath: "persistence-save"
            ),
            BenchmarkReport.Row(
                label: "persistence=load",
                recallAt10: 0,
                qps: 0,
                buildTimeMs: buildMs,
                p50Ms: 0,
                p95Ms: 0,
                p99Ms: 0,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "load",
                operationTimeMs: loadMs,
                estimatedBackendPath: "persistence-load"
            ),
            row(
                label: "persistence=first-query",
                stats: firstStats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "first-query",
                operationTimeMs: firstQueryMs,
                estimatedBackendPath: "cold-loaded"
            ),
            row(
                label: "persistence=warm-query",
                stats: warmStats,
                buildTimeMs: buildMs,
                requestedK: config.k,
                effectiveK: effectiveK,
                operation: "warm-query",
                estimatedBackendPath: "warm-loaded"
            )
        ])
    }

    private static func measureSearches(
        queries: [[Float]],
        expected: [[UInt32]],
        repeatRuns: Int,
        warmupRuns: Int,
        search: ([Float]) async throws -> [SearchResult]
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
        latencies.reserveCapacity(queries.count * repeats)
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

    private static func row(
        label: String,
        stats: SearchStats,
        buildTimeMs: Double,
        requestedK: Int,
        effectiveK: Int,
        operation: String = "query",
        operationTimeMs: Double = 0,
        estimatedBackendPath: String
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
            estimatedBackendPath: estimatedBackendPath
        )
    }

    private static func exactNeighbors(
        queries: [[Float]],
        vectors: [[Float]],
        allowedIDs: Set<String>,
        k: Int,
        metric: Metric
    ) -> [[UInt32]] {
        guard !allowedIDs.isEmpty else {
            return Array(repeating: [], count: queries.count)
        }
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

    private static func matchingIDs(total: Int, ratio: Double) -> [String] {
        let count = max(1, min(total, Int((Double(total) * ratio).rounded())))
        guard count < total else {
            return ids(count: total)
        }

        var selected: [String] = []
        selected.reserveCapacity(count)
        var used = Set<Int>()
        for offset in 0..<count {
            var index = Int((Double(offset) * Double(total) / Double(count)).rounded(.down))
            while used.contains(index), index + 1 < total {
                index += 1
            }
            used.insert(index)
            selected.append("v_\(index)")
        }
        return selected
    }

    private static func deletionIDs(total: Int, ratio: Double) -> [String] {
        guard total > 1 else {
            return []
        }
        let requestedCount = Int((Double(total) * ratio).rounded())
        let count = max(1, min(total - 1, requestedCount))
        return matchingIDs(total: total, ratio: Double(count) / Double(total))
    }

    static func ratioLabel(_ ratio: Double) -> String {
        String(format: "%.3f", ratio)
    }

    private static func percentile(_ p: Double, in values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        let rank = Int(ceil(p * Double(values.count))) - 1
        let index = min(max(rank, 0), values.count - 1)
        return values[index]
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
}

private extension Array where Element == [UInt32] {
    func maxNeighborCount() -> Int {
        reduce(into: 0) { count, row in
            count = Swift.max(count, row.count)
        }
    }
}
