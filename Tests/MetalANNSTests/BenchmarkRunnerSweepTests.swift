import Testing
@testable import MetalANNSBenchmarks
import MetalANNSCore

@Suite("BenchmarkRunner Sweep Tests")
struct BenchmarkRunnerSweepTests {
    @Test("parserAcceptsScaleKnobs")
    func parserAcceptsScaleKnobs() throws {
        let options = try parseOptions(from: [
            "--vector-count", "10000",
            "--dimension", "384",
            "--query-count", "25"
        ])

        #expect(options.vectorCount == 10_000)
        #expect(options.dimension == 384)
        #expect(options.queryCount == 25)
    }

    @Test("parserAcceptsLifecycleWorkloadModes")
    func parserAcceptsLifecycleWorkloadModes() throws {
        let filterOptions = try parseOptions(from: [
            "--filter-sweep",
            "--filter-selectivity", "0.5,0.1,0.01"
        ])
        let deleteOptions = try parseOptions(from: [
            "--delete-sweep",
            "--delete-ratios", "0.1,0.5"
        ])
        let persistenceOptions = try parseOptions(from: ["--persistence"])

        #expect(filterOptions.mode == .filterSweep)
        #expect(filterOptions.filterSelectivities == [0.5, 0.1, 0.01])
        #expect(deleteOptions.mode == .deleteSweep)
        #expect(deleteOptions.deleteRatios == [0.1, 0.5])
        #expect(persistenceOptions.mode == .persistence)
    }

    @Test("parserAcceptsTopTierBenchmarkModes")
    func parserAcceptsTopTierBenchmarkModes() throws {
        let storageOptions = try parseOptions(from: [
            "--storage-sweep",
            "--storage-modes", "normal,mmap,disk-backed"
        ])
        let streamingOptions = try parseOptions(from: [
            "--streaming",
            "--streaming-batch-size", "16"
        ])
        let shardedOptions = try parseOptions(from: [
            "--sharded",
            "--shards", "4",
            "--nprobe", "2"
        ])
        let concurrentOptions = try parseOptions(from: [
            "--concurrent",
            "--concurrency", "3"
        ])
        let usearchOptions = try parseOptions(from: ["--usearch-compare"])

        #expect(storageOptions.mode == .storageSweep)
        #expect(storageOptions.storageModes == [.normal, .mmap, .diskBacked])
        #expect(streamingOptions.mode == .streaming)
        #expect(streamingOptions.streamingBatchSize == 16)
        #expect(shardedOptions.mode == .sharded)
        #expect(shardedOptions.shards == 4)
        #expect(shardedOptions.nprobe == 2)
        #expect(concurrentOptions.mode == .concurrent)
        #expect(concurrentOptions.concurrency == 3)
        #expect(usearchOptions.mode == .usearchCompare)
    }

    @Test("parserRejectsInvalidRatioLists")
    func parserRejectsInvalidRatioLists() {
        #expect(throws: BenchmarkDatasetError.self) {
            _ = try parseOptions(from: ["--filter-sweep", "--filter-selectivity", "0.5,bad"])
        }
        #expect(throws: BenchmarkDatasetError.self) {
            _ = try parseOptions(from: ["--delete-sweep", "--delete-ratios", "0,1.2"])
        }
    }

    @Test("parserRejectsInvalidSweepValues")
    func parserRejectsInvalidSweepValues() {
        #expect(throws: BenchmarkDatasetError.self) {
            _ = try parseOptions(from: ["--sweep", "--sweep-efsearch", "16,bad,64"])
        }
    }

    @Test("swiftPMTestRunnerInvocationCanBeSkipped")
    func swiftPMTestRunnerInvocationCanBeSkipped() {
        #expect(shouldSkipBenchmarkExecution(for: ["--test-bundle-path", "/tmp/tests.xctest"]))
        #expect(!shouldSkipBenchmarkExecution(for: ["--query-count", "5"]))
    }

    @Test("syntheticConfigUsesScaleKnobs")
    func syntheticConfigUsesScaleKnobs() throws {
        let options = try parseOptions(from: [
            "--vector-count", "1234",
            "--dimension", "96",
            "--query-count", "12",
            "--k", "10"
        ])
        let config = syntheticBaseConfig(from: options, defaultMetric: .cosine)

        #expect(config.vectorCount == 1_234)
        #expect(config.dim == 96)
        #expect(config.queryCount == 12)
        #expect(config.k == 10)
    }

    @Test("deleteSweepKeepsTinyCorporaQueryable")
    func deleteSweepKeepsTinyCorporaQueryable() async throws {
        let config = BenchmarkRunner.Config(
            vectorCount: 2,
            dim: 8,
            degree: 1,
            queryCount: 1,
            k: 1,
            efSearch: 4,
            metric: .cosine
        )

        let result = try await LifecycleBenchmark.deleteSweep(
            config: config,
            deleteRatios: [0.9],
            repeatRuns: 1,
            warmupRuns: 0,
            seed: 42
        )

        #expect(result.rows.count == 2)
        #expect(result.rows.allSatisfy { $0.queryCount == 1 })
        #expect(result.rows.allSatisfy { $0.buildTimeMs > 0 })
        #expect(result.rows.allSatisfy { $0.operationTimeMs >= 0 })
        #expect(result.rows.map(\.operation) == ["delete", "compact"])
    }

    @Test("datasetFingerprintCoversAllVectorsAndGroundTruth")
    func datasetFingerprintCoversAllVectorsAndGroundTruth() {
        let base = BenchmarkDataset(
            trainVectors: [[1, 2], [3, 4]],
            testVectors: [[5, 6], [7, 8]],
            groundTruth: [[0, 1], [1, 0]],
            dimension: 2,
            metric: .cosine,
            neighborsCount: 2
        )
        let changedTrain = BenchmarkDataset(
            trainVectors: [[1, 2], [3, 40]],
            testVectors: base.testVectors,
            groundTruth: base.groundTruth,
            dimension: base.dimension,
            metric: base.metric,
            neighborsCount: base.neighborsCount
        )
        let changedTruth = BenchmarkDataset(
            trainVectors: base.trainVectors,
            testVectors: base.testVectors,
            groundTruth: [[0, 1], [0, 0]],
            dimension: base.dimension,
            metric: base.metric,
            neighborsCount: base.neighborsCount
        )

        let baseHash = datasetFingerprint(base)

        #expect(baseHash.count == 64)
        #expect(datasetFingerprint(changedTrain) != baseHash)
        #expect(datasetFingerprint(changedTruth) != baseHash)
    }

    @Test("advancedBenchmarkModesReturnMeasuredRows")
    func advancedBenchmarkModesReturnMeasuredRows() async throws {
        let config = BenchmarkRunner.Config(
            vectorCount: 48,
            dim: 8,
            degree: 4,
            queryCount: 4,
            k: 4,
            efSearch: 16,
            metric: .cosine
        )

        let storage = try await AdvancedBenchmark.storageSweep(
            config: config,
            modes: [.normal],
            repeatRuns: 1,
            warmupRuns: 0,
            seed: 42
        )
        let streaming = try await AdvancedBenchmark.streaming(
            config: config,
            batchSize: 8,
            repeatRuns: 1,
            warmupRuns: 0,
            seed: 42
        )
        let sharded = try await AdvancedBenchmark.sharded(
            config: config,
            shards: 2,
            nprobe: 1,
            repeatRuns: 1,
            warmupRuns: 0,
            seed: 42
        )
        let concurrent = try await AdvancedBenchmark.concurrent(
            config: config,
            concurrency: 2,
            repeatRuns: 1,
            warmupRuns: 0,
            seed: 42
        )

        #expect(storage.rows.contains { $0.label == "storage=normal-load" && $0.fileSizeBytes > 0 })
        #expect(streaming.rows.contains { $0.label == "streaming=warm-query" && $0.queryCount == 4 })
        #expect(sharded.rows.contains { $0.label == "sharded=search" && $0.queryCount == 4 })
        #expect(concurrent.rows.contains { $0.label == "concurrent=search" && $0.queryCount == 4 })
        #expect(AdvancedBenchmark.unavailableComparatorRows(kind: "usearch")[0].operation == "usearch-unavailable")
    }

    @Test("sweepReturnsOneRowPerConfig")
    func sweepReturnsOneRowPerConfig() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 200,
            testCount: 50,
            dimension: 32,
            k: 100,
            metric: .cosine,
            seed: 42
        )

        let configs: [(label: String, config: BenchmarkRunner.Config)] = [
            ("efSearch=16", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 16, metric: .cosine)),
            ("efSearch=64", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 64, metric: .cosine)),
            ("efSearch=128", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 128, metric: .cosine))
        ]

        let report = try await BenchmarkRunner.sweep(configs: configs, dataset: dataset)
        #expect(report.rows.count == 3)
    }

    @Test("qpsIsPositive")
    func qpsIsPositive() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 200,
            testCount: 50,
            dimension: 32,
            k: 100,
            metric: .cosine,
            seed: 123
        )

        let configs: [(label: String, config: BenchmarkRunner.Config)] = [
            ("efSearch=32", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 32, metric: .cosine)),
            ("efSearch=64", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 64, metric: .cosine)),
            ("efSearch=128", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 128, metric: .cosine))
        ]

        let report = try await BenchmarkRunner.sweep(configs: configs, dataset: dataset)
        #expect(report.rows.allSatisfy { $0.qps > 0 })
    }

    @Test("recallFromDataset")
    func recallFromDataset() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 200,
            testCount: 50,
            dimension: 32,
            k: 100,
            metric: .cosine,
            seed: 9
        )

        let config = BenchmarkRunner.Config(
            vectorCount: 200,
            dim: 32,
            degree: 32,
            queryCount: 50,
            k: 10,
            efSearch: 128,
            metric: .cosine
        )

        let results = try await BenchmarkRunner.run(config: config, dataset: dataset)
        #expect(results.recallAt10 > 0.5)
    }
}
