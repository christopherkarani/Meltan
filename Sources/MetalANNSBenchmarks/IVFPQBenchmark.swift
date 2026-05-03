import Foundation
import MetalANNS
import MetalANNSCore

public struct IVFPQBenchmark: Sendable {
    public struct ComparisonResults: Sendable {
        public var annsResults: BenchmarkReport.Row
        public var ivfpqResults: BenchmarkReport.Row

        public init(annsResults: BenchmarkReport.Row, ivfpqResults: BenchmarkReport.Row) {
            self.annsResults = annsResults
            self.ivfpqResults = ivfpqResults
        }
    }

    static func run(
        dataset: BenchmarkDataset,
        annsConfig: BenchmarkRunner.Config,
        ivfpqConfig: IVFPQConfiguration,
        queryCount: Int? = nil,
        repeatRuns: Int = 1,
        warmupRuns: Int = 0
    ) async throws -> ComparisonResults {
        let repeats = max(1, repeatRuns)
        let warmups = max(0, warmupRuns)
        guard !dataset.testVectors.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("dataset contains no query vectors")
        }
        guard !dataset.groundTruth.isEmpty else {
            throw BenchmarkDatasetError.invalidDataset("dataset contains no ground truth")
        }

        let requestedQueryCount = max(1, queryCount ?? annsConfig.queryCount)
        let effectiveQueryCount = min(requestedQueryCount, dataset.testVectors.count)
        guard effectiveQueryCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("queryCount exceeds available query count")
        }
        guard dataset.groundTruth.count >= effectiveQueryCount else {
            throw BenchmarkDatasetError.invalidDataset("groundTruth does not contain enough rows")
        }

        let queries = Array(dataset.testVectors.prefix(effectiveQueryCount))
        let expectedNeighbors = Array(dataset.groundTruth.prefix(effectiveQueryCount))

        let benchmarkDataset = BenchmarkDataset(
            trainVectors: dataset.trainVectors,
            testVectors: queries,
            groundTruth: expectedNeighbors,
            dimension: dataset.dimension,
            metric: dataset.metric,
            neighborsCount: dataset.neighborsCount
        )

        var adjustedAnnsConfig = annsConfig
        adjustedAnnsConfig.queryCount = benchmarkDataset.testVectors.count

        let anns = try await BenchmarkRunner.run(
            config: adjustedAnnsConfig,
            dataset: benchmarkDataset,
            repeatRuns: repeats,
            warmupRuns: warmups
        )
        let annsRow = BenchmarkReport.Row(
            label: "_GraphIndex",
            recallAt10: anns.recallAt10,
            qps: anns.qps,
            buildTimeMs: anns.buildTimeMs,
            p50Ms: anns.queryLatencyP50Ms,
            p95Ms: anns.queryLatencyP95Ms,
            p99Ms: anns.queryLatencyP99Ms,
            recallAt1: anns.recallAt1,
            recallAt100: anns.recallAt100,
            queryCount: anns.queryCount,
            avgQueryMs: anns.queryLatencyMeanMs,
            maxQueryMs: anns.queryLatencyMaxMs,
            p90Ms: anns.latencyDistribution.p90Ms,
            p999Ms: anns.latencyDistribution.p999Ms,
            stdDevMs: anns.queryLatencyStdDevMs,
            minMs: anns.queryLatencyMinMs,
            indexResidentMB: Double(anns.indexResidentBytesEstimate) / (1024 * 1024),
            peakResidentMB: anns.memoryAfterQueries.peakResidentMB
        )

        let capacity = max(dataset.trainVectors.count * 2, dataset.trainVectors.count + 1)
        let ivfpqIndex = try _IVFPQIndex(
            capacity: capacity,
            dimension: dataset.dimension,
            config: ivfpqConfig
        )

        let ivfpqMemoryBeforeBuild = MemorySnapshot.capture()
        let trainStart = DispatchTime.now().uptimeNanoseconds
        try await ivfpqIndex.train(vectors: dataset.trainVectors)
        let ids = (0..<dataset.trainVectors.count).map { "v_\($0)" }
        try await ivfpqIndex.add(vectors: dataset.trainVectors, ids: ids)
        let trainEnd = DispatchTime.now().uptimeNanoseconds
        let buildTimeMs = Double(trainEnd - trainStart) / 1_000_000.0
        let ivfpqMemoryAfterBuild = MemorySnapshot.capture()

        let top1Count = min(1, dataset.neighborsCount)
        let top10Count = min(10, dataset.neighborsCount)
        let top100Count = min(100, dataset.neighborsCount)
        let queryK = max(10, top100Count)

        var allLatencies: [Double] = []
        allLatencies.reserveCapacity(queries.count * repeats)

        var recallAt1Total: Double = 0
        var recallAt10Total: Double = 0
        var recallAt100Total: Double = 0
        var totalSearchTimeSeconds: Double = 0

        if warmups > 0 {
            for _ in 0..<warmups {
                _ = try await benchmarkIVFPQBatch(
                    index: ivfpqIndex,
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

        for _ in 0..<repeats {
            let batchStats = try await benchmarkIVFPQBatch(
                index: ivfpqIndex,
                queryK: queryK,
                top1Count: top1Count,
                top10Count: top10Count,
                top100Count: top100Count,
                queries: queries,
                expectedNeighbors: expectedNeighbors,
                measureLatency: true
            )
            allLatencies.append(contentsOf: batchStats.latencies)
            recallAt1Total += batchStats.recallAt1
            recallAt10Total += batchStats.recallAt10
            recallAt100Total += batchStats.recallAt100
            totalSearchTimeSeconds += batchStats.totalSearchSeconds
        }

        let totalQueryCount = queries.count * repeats
        let ivfpqDistribution = LatencyDistribution.compute(fromLatenciesMs: allLatencies)
        let ivfpqMemoryAfterQueries = MemorySnapshot.capture()
        let ivfpqResidentDelta = ivfpqMemoryAfterBuild.residentBytes
            &- min(ivfpqMemoryAfterBuild.residentBytes, ivfpqMemoryBeforeBuild.residentBytes)
        let ivfpqRow = BenchmarkReport.Row(
            label: "_IVFPQIndex",
            recallAt10: totalQueryCount > 0 ? recallAt10Total / Double(totalQueryCount) : 0,
            qps: totalQueryCount > 0 && totalSearchTimeSeconds > 0 ? Double(totalQueryCount) / totalSearchTimeSeconds : 0,
            buildTimeMs: buildTimeMs,
            p50Ms: ivfpqDistribution.p50Ms,
            p95Ms: ivfpqDistribution.p95Ms,
            p99Ms: ivfpqDistribution.p99Ms,
            recallAt1: totalQueryCount > 0 ? recallAt1Total / Double(totalQueryCount) : 0,
            recallAt100: totalQueryCount > 0 ? recallAt100Total / Double(totalQueryCount) : 0,
            queryCount: totalQueryCount,
            avgQueryMs: ivfpqDistribution.meanMs,
            maxQueryMs: ivfpqDistribution.maxMs,
            p90Ms: ivfpqDistribution.p90Ms,
            p999Ms: ivfpqDistribution.p999Ms,
            stdDevMs: ivfpqDistribution.stdDevMs,
            minMs: ivfpqDistribution.minMs,
            indexResidentMB: Double(ivfpqResidentDelta) / (1024 * 1024),
            peakResidentMB: ivfpqMemoryAfterQueries.peakResidentMB
        )

        return ComparisonResults(annsResults: annsRow, ivfpqResults: ivfpqRow)
    }

    static func renderComparison(_ results: ComparisonResults) -> String {
        BenchmarkReport(
            rows: [results.annsResults, results.ivfpqResults],
            datasetLabel: "comparison"
        ).renderTable()
    }

    private static func benchmarkIVFPQBatch(
        index: _IVFPQIndex,
        queryK: Int,
        top1Count: Int,
        top10Count: Int,
        top100Count: Int,
        queries: [[Float]],
        expectedNeighbors: [[UInt32]],
        measureLatency: Bool
    ) async throws -> IVFPQBenchmarkBatchStats {
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
            let results = await index.search(query: query, k: queryK)
            let searchEnd = DispatchTime.now().uptimeNanoseconds

            if measureLatency {
                latencies.append(Double(searchEnd - searchStart) / 1_000_000.0)
            }

            let approxTop1 = BenchmarkIDParser.uint32Set(from: results, limit: top1Count)
            let approxTop10 = BenchmarkIDParser.uint32Set(from: results, limit: top10Count)
            let approxTop100 = BenchmarkIDParser.uint32Set(from: results, limit: top100Count)

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

        return IVFPQBenchmarkBatchStats(
            latencies: latencies,
            recallAt1: recallAt1Total,
            recallAt10: recallAt10Total,
            recallAt100: recallAt100Total,
            totalSearchSeconds: Double(batchEnd - batchStart) / 1_000_000_000.0
        )
    }

    private struct IVFPQBenchmarkBatchStats {
        let latencies: [Double]
        let recallAt1: Double
        let recallAt10: Double
        let recallAt100: Double
        let totalSearchSeconds: Double
    }
}
