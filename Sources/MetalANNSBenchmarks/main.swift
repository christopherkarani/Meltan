import Foundation
import CryptoKit
import Metal
import MetalANNS
import MetalANNSCore

let defaultSweepValues = [16, 32, 64, 128, 256]

let args = Array(CommandLine.arguments.dropFirst())

func reportRow(from results: BenchmarkRunner.Results, label: String = "single") -> BenchmarkReport.Row {
    BenchmarkReport.Row(
        label: label,
        recallAt10: results.recallAt10,
        qps: results.qps,
        buildTimeMs: results.buildTimeMs,
        p50Ms: results.queryLatencyP50Ms,
        p95Ms: results.queryLatencyP95Ms,
        p99Ms: results.queryLatencyP99Ms,
        recallAt1: results.recallAt1,
        recallAt100: results.recallAt100,
        queryCount: results.queryCount,
        avgQueryMs: results.queryLatencyMeanMs,
        maxQueryMs: results.queryLatencyMaxMs,
        requestedK: results.requestedK,
        effectiveK: results.effectiveK,
        estimatedBackendPath: results.estimatedBackendPath
    )
}

func benchmarkMetadata(
    mode: String,
    config: BenchmarkRunner.Config,
    datasetLabel: String,
    csvOut: String?,
    repeatRuns: Int,
    warmupRuns: Int,
    seed: Int?,
    datasetHash: String
) -> [String: String] {
    [
        "mode": mode,
        "datasetLabel": datasetLabel,
        "datasetHash": datasetHash,
        "vectorCount": String(config.vectorCount),
        "queryCount": String(config.queryCount),
        "dim": String(config.dim),
        "degree": String(config.degree),
        "k": String(config.k),
        "efSearch": String(config.efSearch),
        "metric": config.metric.rawValue,
        "runs": String(repeatRuns),
        "warmupRuns": String(warmupRuns),
        "swiftCompilerVersion": swiftCompilerVersion(),
        "osVersion": ProcessInfo.processInfo.operatingSystemVersionString,
        "gitCommit": gitCommit(),
        "machine": hostMachine(),
        "processorCount": String(ProcessInfo.processInfo.processorCount),
        "activeProcessorCount": String(ProcessInfo.processInfo.activeProcessorCount),
        "physicalMemoryBytes": String(ProcessInfo.processInfo.physicalMemory),
        "metalDevice": MTLCreateSystemDefaultDevice()?.name ?? "unavailable",
        "generatedAt": ISO8601DateFormatter().string(from: Date()),
        "csvOut": csvOut ?? "",
        "seed": seed != nil ? String(seed!) : ""
    ]
}

func makeBenchmarkConfig(
    from base: BenchmarkRunner.Config,
    _ options: ParsedBenchmarkOptions
) throws -> BenchmarkRunner.Config {
    var config = base

    if let queryCount = options.queryCount {
        config.queryCount = max(1, queryCount)
    }
    if let degree = options.degree {
        config.degree = max(1, degree)
    }
    if let efSearch = options.efSearch {
        config.efSearch = max(1, efSearch)
    }
    if let k = options.k {
        config.k = max(1, k)
    }
    if let metric = options.metric {
        config.metric = metric
    }

    return config
}

func syntheticBaseConfig(
    from options: ParsedBenchmarkOptions,
    defaultMetric: Metric
) -> BenchmarkRunner.Config {
    BenchmarkRunner.Config(
        vectorCount: options.vectorCount ?? 1000,
        dim: options.dimension ?? 128,
        queryCount: options.queryCount ?? 100,
        k: options.k ?? 10,
        efSearch: options.efSearch ?? 64,
        metric: options.metric ?? defaultMetric
    )
}

func loadOrSyntheticDataset(
    path: String?,
    baseConfig: BenchmarkRunner.Config,
    metric: Metric,
    seed: Int
) throws -> (dataset: BenchmarkDataset, source: String) {
    if let path {
        let dataset = try BenchmarkDataset.load(from: path)
        return (dataset, path)
    }

    let dataset = BenchmarkDataset.synthetic(
        trainCount: baseConfig.vectorCount,
        testCount: baseConfig.queryCount,
        dimension: baseConfig.dim,
        k: max(100, baseConfig.k),
        metric: metric,
        seed: seed
    )

    return (dataset, "synthetic")
}

func printResults(_ results: BenchmarkRunner.Results) {
    print("Build time:          \(String(format: "%.1f", results.buildTimeMs)) ms")
    print("Query count:         \(results.queryCount)")
    print("Requested k:         \(results.requestedK)")
    print("Effective search k:  \(results.effectiveK)")
    print("Estimated backend:   \(results.estimatedBackendPath)")
    print("Query mean:          \(String(format: "%.3f", results.queryLatencyMeanMs)) ms")
    print("Query p50:           \(String(format: "%.2f", results.queryLatencyP50Ms)) ms")
    print("Query p95:           \(String(format: "%.2f", results.queryLatencyP95Ms)) ms")
    print("Query p99:           \(String(format: "%.2f", results.queryLatencyP99Ms)) ms")
    print("Query stddev:        \(String(format: "%.3f", results.queryLatencyStdDevMs)) ms")
    print("Query QPS:           \(String(format: "%.2f", results.qps))")
    print("Recall@1:            \(String(format: "%.3f", results.recallAt1))")
    print("Recall@10:           \(String(format: "%.3f", results.recallAt10))")
    print("Recall@100:          \(String(format: "%.3f", results.recallAt100))")
}

func makeConfigsForSweep(
    from base: BenchmarkRunner.Config,
    values: [Int],
    dataset: BenchmarkDataset
) -> [(label: String, config: BenchmarkRunner.Config)] {
    values.map { efSearch in
        (
            label: "efSearch=\(efSearch)",
            config: BenchmarkRunner.Config(
                vectorCount: dataset.trainVectors.count,
                dim: dataset.dimension,
                degree: base.degree,
                queryCount: base.queryCount,
                k: base.k,
                efSearch: efSearch,
                metric: base.metric
            )
        )
    }
}

func printUsage() {
    print("USAGE:")
    print("  MetalANNSBenchmarks [--dataset <path.annbin>]                                  # synthetic or loaded single run (default)")
    print("  MetalANNSBenchmarks --sweep [--dataset <path.annbin>]                          # sweep efSearch")
    print("  MetalANNSBenchmarks --dataset <path.annbin> --sweep [--sweep-efsearch <list>]  # dataset-aware sweep")
    print("  MetalANNSBenchmarks --dataset <path.annbin> --csv-out <path.csv>                # save CSV")
    print("  MetalANNSBenchmarks --ivfpq                                                    # ANS vs IVFPQ (synthetic if no dataset)")
    print("  MetalANNSBenchmarks --filter-sweep                                             # filtered exact-ground-truth sweep")
    print("  MetalANNSBenchmarks --delete-sweep                                             # soft-delete and compact sweep")
    print("  MetalANNSBenchmarks --persistence                                              # save/load/cold-query benchmark")
    print("  MetalANNSBenchmarks --storage-sweep                                            # normal/mmap/disk-backed load benchmark")
    print("  MetalANNSBenchmarks --streaming                                                # sustained ingest/search/merge benchmark")
    print("  MetalANNSBenchmarks --sharded                                                  # sharded build/search benchmark")
    print("  MetalANNSBenchmarks --concurrent                                               # concurrent search and mixed read/write benchmark")
    print("  MetalANNSBenchmarks --gpu-parity                                               # GPU parity availability check")
    print("  MetalANNSBenchmarks --usearch-compare                                          # USearch comparator availability check")
    print("\nOPTIONS:")
    print("  --query-count <n>        override number of query vectors")
    print("  --vector-count <n>       synthetic train vector count")
    print("  --dimension <n>          synthetic vector dimension")
    print("  --seed <n>               deterministic seed for synthetic dataset")
    print("  --runs <n>               number of measured benchmark passes")
    print("  --warmup <n>             number of warmup passes")
    print("  --sweep-efsearch <list>  comma-separated ef values (default: 16,32,64,128,256)")
    print("  --filter-selectivity <list> comma-separated ratios in (0,1] (default: 0.5,0.1,0.01)")
    print("  --delete-ratios <list>   comma-separated ratios in (0,1) (default: 0.1,0.5)")
    print("  --storage-modes <list>   comma-separated normal,mmap,disk-backed")
    print("  --streaming-batch-size <n>")
    print("  --shards <n>")
    print("  --nprobe <n>")
    print("  --concurrency <n>")
    print("  --dataset <path.annbin>   use real dataset")
    print("  --csv-out <path.csv>      save CSV report")
    print("  --json-out <path.json>    save JSON report")
    print("  --metric <cosine|l2|innerproduct|hamming>")
    print("  --degree <n>")
    print("  --efsearch <n>")
    print("  --k <n>")
    print("  --ivfpq-subspaces <n>")
    print("  --ivfpq-centroids <n>")
    print("  --ivfpq-coarse-centroids <n>")
    print("  --ivfpq-nprobe <n>")
    print("  --ivfpq-iterations <n>")
    print("  --help")
}

struct ParsedBenchmarkOptions {
    enum Mode {
        case single
        case sweep
        case ivfpq
        case filterSweep
        case deleteSweep
        case persistence
        case storageSweep
        case streaming
        case sharded
        case concurrent
        case gpuParity
        case usearchCompare
        case help
    }

    var mode: Mode = .single
    var shouldPrintUsage = false
    var datasetPath: String?
    var csvOutPath: String?
    var jsonOutPath: String?
    var queryCount: Int?
    var vectorCount: Int?
    var dimension: Int?
    var seed: Int? = 42
    var repeatRuns: Int = 1
    var warmupRuns: Int = 0
    var sweepEfSearchValues: [Int] = []
    var filterSelectivities: [Double] = []
    var deleteRatios: [Double] = []
    var storageModes: [StorageBenchmarkMode] = []
    var streamingBatchSize: Int = 64
    var shards: Int = 4
    var nprobe: Int = 2
    var concurrency: Int = 4
    var metric: Metric?
    var degree: Int?
    var efSearch: Int?
    var k: Int?

    var ivfpqSubspaces: Int = 8
    var ivfpqNumCentroids: Int = 256
    var ivfpqNumCoarseCentroids: Int = 256
    var ivfpqNprobe: Int = 8
    var ivfpqTrainingIterations: Int = 10
}

enum StorageBenchmarkMode: String, Sendable, Equatable {
    case normal
    case mmap
    case diskBacked = "disk-backed"
}

func parseOptions(from args: [String]) throws -> ParsedBenchmarkOptions {
    var options = ParsedBenchmarkOptions()
    var index = 0

    while index < args.count {
        let arg = args[index]

        switch arg {
        case "--help", "-h":
            options.shouldPrintUsage = true
            options.mode = .help

        case "--sweep":
            options.mode = .sweep

        case "--ivfpq":
            options.mode = .ivfpq

        case "--filter-sweep":
            options.mode = .filterSweep

        case "--delete-sweep":
            options.mode = .deleteSweep

        case "--persistence":
            options.mode = .persistence

        case "--storage-sweep":
            options.mode = .storageSweep

        case "--streaming":
            options.mode = .streaming

        case "--sharded":
            options.mode = .sharded

        case "--concurrent":
            options.mode = .concurrent

        case "--gpu-parity":
            options.mode = .gpuParity

        case "--usearch-compare":
            options.mode = .usearchCompare

        case "--dataset":
            options.datasetPath = try nextValue(for: arg, args: args, index: &index)

        case "--csv-out":
            options.csvOutPath = try nextValue(for: arg, args: args, index: &index)

        case "--json-out":
            options.jsonOutPath = try nextValue(for: arg, args: args, index: &index)

        case "--query-count":
            let value = try nextValue(for: arg, args: args, index: &index)
            guard let parsed = Int(value), parsed > 0 else {
                throw BenchmarkDatasetError.invalidDataset("Invalid --query-count value: \(value)")
            }
            options.queryCount = parsed

        case "--vector-count":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.vectorCount = try parsePositiveInt(arg, value)

        case "--dimension":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.dimension = try parsePositiveInt(arg, value)

        case "--seed":
            let value = try nextValue(for: arg, args: args, index: &index)
            guard let parsed = Int(value) else {
                throw BenchmarkDatasetError.invalidDataset("Invalid --seed value: \(value)")
            }
            options.seed = parsed

        case "--runs":
            let value = try nextValue(for: arg, args: args, index: &index)
            guard let parsed = Int(value), parsed > 0 else {
                throw BenchmarkDatasetError.invalidDataset("Invalid --runs value: \(value)")
            }
            options.repeatRuns = parsed

        case "--warmup":
            let value = try nextValue(for: arg, args: args, index: &index)
            guard let parsed = Int(value), parsed >= 0 else {
                throw BenchmarkDatasetError.invalidDataset("Invalid --warmup value: \(value)")
            }
            options.warmupRuns = parsed

        case "--sweep-efsearch":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.sweepEfSearchValues = try parseIntList(value)

        case "--filter-selectivity":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.filterSelectivities = try parseRatioList(
                value,
                flag: arg,
                upperBound: .inclusive
            )

        case "--delete-ratios":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.deleteRatios = try parseRatioList(
                value,
                flag: arg,
                upperBound: .exclusive
            )

        case "--storage-modes":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.storageModes = try parseStorageModes(value)

        case "--streaming-batch-size":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.streamingBatchSize = try parsePositiveInt(arg, value)

        case "--shards":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.shards = try parsePositiveInt(arg, value)

        case "--nprobe":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.nprobe = try parsePositiveInt(arg, value)

        case "--concurrency":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.concurrency = try parsePositiveInt(arg, value)

        case "--metric":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.metric = try parseMetric(value)

        case "--degree":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.degree = try parsePositiveInt(arg, value)

        case "--efsearch":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.efSearch = try parsePositiveInt(arg, value)

        case "--k":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.k = try parsePositiveInt(arg, value)

        case "--ivfpq-subspaces":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.ivfpqSubspaces = try parsePositiveInt(arg, value)

        case "--ivfpq-centroids":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.ivfpqNumCentroids = try parsePositiveInt(arg, value)

        case "--ivfpq-coarse-centroids":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.ivfpqNumCoarseCentroids = try parsePositiveInt(arg, value)

        case "--ivfpq-nprobe":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.ivfpqNprobe = try parsePositiveInt(arg, value)

        case "--ivfpq-iterations":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.ivfpqTrainingIterations = try parsePositiveInt(arg, value)

        default:
            throw BenchmarkDatasetError.invalidDataset("Unknown argument: \(arg)")
        }

        index += 1
    }

    return options
}

func parseMetric(_ value: String) throws -> Metric {
    switch value.lowercased() {
    case "cosine":
        return .cosine
    case "l2":
        return .l2
    case "innerproduct":
        return .innerProduct
    case "hamming":
        return .hamming
    default:
        throw BenchmarkDatasetError.invalidDataset("Unsupported metric: \(value)")
    }
}

func parsePositiveInt(_ flag: String, _ value: String) throws -> Int {
    guard let parsed = Int(value), parsed > 0 else {
        throw BenchmarkDatasetError.invalidDataset("Invalid value for \(flag): \(value)")
    }
    return parsed
}

func parseIntList(_ value: String) throws -> [Int] {
    let parts = value
        .split(separator: ",", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

    if parts.isEmpty {
        throw BenchmarkDatasetError.invalidDataset("Invalid list for --sweep-efsearch: \(value)")
    }

    var values: [Int] = []
    values.reserveCapacity(parts.count)
    for part in parts {
        guard let parsed = Int(part), parsed > 0 else {
            throw BenchmarkDatasetError.invalidDataset("Invalid list for --sweep-efsearch: \(value)")
        }
        values.append(parsed)
    }

    return values
}

enum RatioUpperBound {
    case inclusive
    case exclusive
}

func parseRatioList(
    _ value: String,
    flag: String,
    upperBound: RatioUpperBound
) throws -> [Double] {
    let parts = value
        .split(separator: ",", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

    if parts.isEmpty {
        throw BenchmarkDatasetError.invalidDataset("Invalid list for \(flag): \(value)")
    }

    var values: [Double] = []
    values.reserveCapacity(parts.count)
    for part in parts {
        guard let parsed = Double(part), parsed.isFinite, parsed > 0 else {
            throw BenchmarkDatasetError.invalidDataset("Invalid list for \(flag): \(value)")
        }
        switch upperBound {
        case .inclusive:
            guard parsed <= 1 else {
                throw BenchmarkDatasetError.invalidDataset("Invalid list for \(flag): \(value)")
            }
        case .exclusive:
            guard parsed < 1 else {
                throw BenchmarkDatasetError.invalidDataset("Invalid list for \(flag): \(value)")
            }
        }
        values.append(parsed)
    }

    return values
}

func parseStorageModes(_ value: String) throws -> [StorageBenchmarkMode] {
    let parts = value
        .split(separator: ",", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }

    if parts.isEmpty {
        throw BenchmarkDatasetError.invalidDataset("Invalid list for --storage-modes: \(value)")
    }

    var modes: [StorageBenchmarkMode] = []
    modes.reserveCapacity(parts.count)
    for part in parts {
        guard let mode = StorageBenchmarkMode(rawValue: part) else {
            throw BenchmarkDatasetError.invalidDataset("Invalid list for --storage-modes: \(value)")
        }
        modes.append(mode)
    }
    return modes
}

func shouldSkipBenchmarkExecution(for args: [String]) -> Bool {
    let swiftPMTestFlags: Set<String> = [
        "--test-bundle-path",
        "--testing-library",
        "--xunit-output"
    ]
    return args.contains { swiftPMTestFlags.contains($0) }
}

func validateDatasetMetric(_ dataset: BenchmarkDataset, options: ParsedBenchmarkOptions) throws {
    if let metric = options.metric, metric != dataset.metric {
        throw BenchmarkDatasetError.invalidDataset(
            "--metric \(metric.rawValue) does not match dataset ground-truth metric \(dataset.metric.rawValue)"
        )
    }
}

func datasetFingerprint(_ dataset: BenchmarkDataset) -> String {
    var data = Data()

    func appendUInt64(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    func appendUInt32(_ value: UInt32) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    func appendInt(_ value: Int) {
        appendUInt64(UInt64(bitPattern: Int64(value)))
    }

    func appendFloat(_ value: Float) {
        appendUInt32(value.bitPattern)
    }

    func appendString(_ value: String) {
        appendInt(value.utf8.count)
        data.append(contentsOf: value.utf8)
    }

    appendInt(dataset.trainVectors.count)
    appendInt(dataset.testVectors.count)
    appendInt(dataset.dimension)
    appendInt(dataset.neighborsCount)
    appendString(dataset.metric.rawValue)

    for vector in dataset.trainVectors {
        appendInt(vector.count)
        for value in vector {
            appendFloat(value)
        }
    }

    for vector in dataset.testVectors {
        appendInt(vector.count)
        for value in vector {
            appendFloat(value)
        }
    }

    for row in dataset.groundTruth {
        appendInt(row.count)
        for value in row {
            appendUInt32(value)
        }
    }

    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func gitCommit() -> String {
    commandOutput("/usr/bin/git", ["rev-parse", "--short=12", "HEAD"]) ?? "unknown"
}

func swiftCompilerVersion() -> String {
    commandOutput("/usr/bin/swift", ["--version"])?
        .split(separator: "\n")
        .first
        .map(String.init) ?? "unknown"
}

func hostMachine() -> String {
    commandOutput("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]) ?? "unknown"
}

func commandOutput(_ executable: String, _ arguments: [String]) -> String? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()

    do {
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    } catch {
        return nil
    }
}

func nextValue(for flag: String, args: [String], index: inout Int) throws -> String {
    let nextIndex = args.index(after: index)
    guard nextIndex < args.count else {
        throw BenchmarkDatasetError.invalidDataset("Missing value for \(flag)")
    }
    index = nextIndex
    return args[index]
}

// --- Benchmark execution ---

do {
    if shouldSkipBenchmarkExecution(for: args) {
        exit(0)
    }

    let options = try parseOptions(from: args)

    if options.shouldPrintUsage {
        printUsage()
        exit(0)
    }

    switch options.mode {
    case .help:
        printUsage()
        exit(0)

    case .single:
        var datasetLabel = "synthetic"
        let dataset: BenchmarkDataset
        if let datasetPath = options.datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
            try validateDatasetMetric(dataset, options: options)
            datasetLabel = datasetPath
        } else {
            let syntheticConfig = syntheticBaseConfig(from: options, defaultMetric: .cosine)
            dataset = BenchmarkDataset.synthetic(
                trainCount: syntheticConfig.vectorCount,
                testCount: syntheticConfig.queryCount,
                dimension: syntheticConfig.dim,
                k: max(100, syntheticConfig.k),
                metric: syntheticConfig.metric,
                seed: options.seed ?? 42
            )
        }

        let baseConfig = try makeBenchmarkConfig(
            from: BenchmarkRunner.Config(
                vectorCount: dataset.trainVectors.count,
                dim: dataset.dimension,
                queryCount: options.queryCount ?? dataset.testVectors.count,
                k: options.k ?? 10,
                efSearch: options.efSearch ?? 64,
                metric: options.metric ?? dataset.metric
            ),
            options
        )

        let results = try await BenchmarkRunner.run(
            config: baseConfig,
            dataset: dataset,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns
        )

        printResults(results)
        print("Dataset: \(datasetLabel)")

        let report = BenchmarkReport(
            rows: [reportRow(from: results)],
            datasetLabel: datasetLabel,
            metadata: benchmarkMetadata(
                mode: "single",
                config: baseConfig,
                datasetLabel: datasetLabel,
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(dataset)
            )
        )

        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .sweep:
        let dataset: BenchmarkDataset
        let datasetLabel: String

        if let datasetPath = options.datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
            try validateDatasetMetric(dataset, options: options)
            datasetLabel = datasetPath
        } else {
            let syntheticConfig = syntheticBaseConfig(from: options, defaultMetric: .cosine)
            let base = try loadOrSyntheticDataset(
                path: nil,
                baseConfig: syntheticConfig,
                metric: syntheticConfig.metric,
                seed: options.seed ?? 42
            )
            dataset = base.dataset
            datasetLabel = "synthetic"
        }

        let base = try makeBenchmarkConfig(
            from: BenchmarkRunner.Config(
                vectorCount: dataset.trainVectors.count,
                dim: dataset.dimension,
                queryCount: options.queryCount ?? dataset.testVectors.count,
                k: options.k ?? 10,
                efSearch: options.efSearch ?? 64,
                metric: options.metric ?? dataset.metric
            ),
            options
        )

        let values = options.sweepEfSearchValues.isEmpty ? defaultSweepValues : options.sweepEfSearchValues
        let configs = makeConfigsForSweep(from: base, values: values, dataset: dataset)

        let report = try await BenchmarkRunner.sweep(
            configs: configs,
            dataset: dataset,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns
        )
        print(report.renderTable())
        print("Pareto frontier points: \(report.paretoFrontier().count)")

        let metadataReport = BenchmarkReport(
            rows: report.rows,
            datasetLabel: report.datasetLabel,
            metadata: benchmarkMetadata(
                mode: "sweep",
                config: base,
                datasetLabel: datasetLabel,
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(dataset)
            )
        )

        if let csvOutPath = options.csvOutPath {
            try metadataReport.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try metadataReport.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .ivfpq:
        let syntheticConfig = syntheticBaseConfig(from: options, defaultMetric: .l2)
        let datasetSource = try loadOrSyntheticDataset(
            path: options.datasetPath,
            baseConfig: syntheticConfig,
            metric: syntheticConfig.metric,
            seed: options.seed ?? 42
        )
        let dataset = datasetSource.dataset
        try validateDatasetMetric(dataset, options: options)
        let datasetLabel = datasetSource.source

        let annsConfig = try makeBenchmarkConfig(
            from: BenchmarkRunner.Config(
                vectorCount: dataset.trainVectors.count,
                dim: dataset.dimension,
                degree: options.degree ?? 32,
                queryCount: options.queryCount ?? dataset.testVectors.count,
                k: options.k ?? 10,
                efSearch: options.efSearch ?? 64,
                metric: dataset.metric
            ),
            options
        )

        let ivfpqConfig = IVFPQConfiguration(
            numSubspaces: options.ivfpqSubspaces,
            numCentroids: options.ivfpqNumCentroids,
            numCoarseCentroids: options.ivfpqNumCoarseCentroids,
            nprobe: options.ivfpqNprobe,
            metric: dataset.metric,
            trainingIterations: options.ivfpqTrainingIterations
        )

        let comparison = try await IVFPQBenchmark.run(
            dataset: dataset,
            annsConfig: annsConfig,
            ivfpqConfig: ivfpqConfig,
            queryCount: options.queryCount,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns
        )

        let report = BenchmarkReport(
            rows: [comparison.annsResults, comparison.ivfpqResults],
            datasetLabel: datasetLabel,
            metadata: benchmarkMetadata(
                mode: "ivfpq",
                config: annsConfig,
                datasetLabel: datasetLabel,
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(dataset)
            )
        )

        print(IVFPQBenchmark.renderComparison(comparison))

        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .filterSweep:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--filter-sweep currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let selectivities = options.filterSelectivities.isEmpty
            ? [0.5, 0.1, 0.01]
            : options.filterSelectivities
        let result = try await LifecycleBenchmark.filterSweep(
            config: config,
            selectivities: selectivities,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "filter-sweep",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .deleteSweep:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--delete-sweep currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let deleteRatios = options.deleteRatios.isEmpty
            ? [0.1, 0.5]
            : options.deleteRatios
        let result = try await LifecycleBenchmark.deleteSweep(
            config: config,
            deleteRatios: deleteRatios,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "delete-sweep",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .persistence:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--persistence currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let result = try await LifecycleBenchmark.persistence(
            config: config,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "persistence",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .storageSweep:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--storage-sweep currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let result = try await AdvancedBenchmark.storageSweep(
            config: config,
            modes: options.storageModes,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "storage-sweep",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .streaming:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--streaming currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let result = try await AdvancedBenchmark.streaming(
            config: config,
            batchSize: options.streamingBatchSize,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "streaming",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .sharded:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--sharded currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let result = try await AdvancedBenchmark.sharded(
            config: config,
            shards: options.shards,
            nprobe: options.nprobe,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "sharded",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .concurrent:
        if options.datasetPath != nil {
            throw BenchmarkDatasetError.invalidDataset("--concurrent currently supports synthetic datasets only")
        }
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        let result = try await AdvancedBenchmark.concurrent(
            config: config,
            concurrency: options.concurrency,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed ?? 42
        )
        let report = BenchmarkReport(
            rows: result.rows,
            datasetLabel: "synthetic",
            metadata: benchmarkMetadata(
                mode: "concurrent",
                config: config,
                datasetLabel: "synthetic",
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed,
                datasetHash: datasetFingerprint(result.dataset)
            )
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .gpuParity:
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        var metadata = benchmarkMetadata(
            mode: "gpu-parity",
            config: config,
            datasetLabel: "synthetic",
            csvOut: options.csvOutPath,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed,
            datasetHash: ""
        )
        metadata["available"] = "false"
        metadata["reason"] = "no public forced GPU/CPU graph search API"
        metadata["nextStep"] = "instrument _GraphIndex actual dispatch path or expose benchmark-only parity hook"
        let report = BenchmarkReport(
            rows: AdvancedBenchmark.unavailableComparatorRows(kind: "gpu-parity"),
            datasetLabel: "synthetic",
            metadata: metadata
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }

    case .usearchCompare:
        let config = try makeBenchmarkConfig(
            from: syntheticBaseConfig(from: options, defaultMetric: .cosine),
            options
        )
        var metadata = benchmarkMetadata(
            mode: "usearch-compare",
            config: config,
            datasetLabel: "synthetic",
            csvOut: options.csvOutPath,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            seed: options.seed,
            datasetHash: ""
        )
        metadata["comparator"] = "USearch"
        metadata["available"] = "false"
        metadata["reason"] = "USearch is not in the package graph"
        metadata["resolutionRisk"] = "USearch currently introduces NumKong/CNumKongDispatch resolution risk"
        let report = BenchmarkReport(
            rows: AdvancedBenchmark.unavailableComparatorRows(kind: "usearch"),
            datasetLabel: "synthetic",
            metadata: metadata
        )
        print(report.renderTable())
        if let csvOutPath = options.csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        if let jsonOutPath = options.jsonOutPath {
            try report.saveJSON(to: jsonOutPath)
            print("Saved JSON: \(jsonOutPath)")
        }
    }
} catch {
    fputs("Benchmark failed: \(error)\n", stderr)
    printUsage()
    exit(1)
}
