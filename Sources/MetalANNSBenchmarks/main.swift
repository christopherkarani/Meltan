import Foundation
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
        p90Ms: results.latencyDistribution.p90Ms,
        p999Ms: results.latencyDistribution.p999Ms,
        stdDevMs: results.queryLatencyStdDevMs,
        minMs: results.queryLatencyMinMs,
        indexResidentMB: Double(results.indexResidentBytesEstimate) / (1024 * 1024),
        peakResidentMB: results.memoryAfterQueries.peakResidentMB,
        concurrency: results.concurrency,
        firstQueryMs: results.firstQueryLatencyMs,
        warmSteadyMeanMs: results.warmSteadyMeanMs
    )
}

func benchmarkMetadata(
    mode: String,
    config: BenchmarkRunner.Config,
    datasetLabel: String,
    csvOut: String?,
    repeatRuns: Int,
    warmupRuns: Int,
    seed: Int?
) -> [String: String] {
    var metadata: [String: String] = [
        "mode": mode,
        "datasetLabel": datasetLabel,
        "vectorCount": String(config.vectorCount),
        "queryCount": String(config.queryCount),
        "dim": String(config.dim),
        "degree": String(config.degree),
        "k": String(config.k),
        "efSearch": String(config.efSearch),
        "metric": config.metric.rawValue,
        "runs": String(repeatRuns),
        "warmupRuns": String(warmupRuns),
        "generatedAt": ISO8601DateFormatter().string(from: Date()),
        "csvOut": csvOut ?? "",
        "seed": seed != nil ? String(seed!) : ""
    ]

    // Merge in environment probe metadata. On key collisions, the env probe wins
    // (it carries the authoritative osVersion, build config, device info, etc.).
    for (key, value) in EnvironmentProbe.capture().asMetadata() {
        metadata[key] = value
    }

    return metadata
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

func printResults(_ results: BenchmarkRunner.Results, environment: EnvironmentProbe? = nil) {
    print("Build time:          \(String(format: "%.1f", results.buildTimeMs)) ms")
    print("Query count:         \(results.queryCount)")
    print("Query mean:          \(String(format: "%.3f", results.queryLatencyMeanMs)) ms")
    print("Query p50:           \(String(format: "%.2f", results.queryLatencyP50Ms)) ms")
    print("Query p90:           \(String(format: "%.2f", results.latencyDistribution.p90Ms)) ms")
    print("Query p95:           \(String(format: "%.2f", results.queryLatencyP95Ms)) ms")
    print("Query p99:           \(String(format: "%.2f", results.queryLatencyP99Ms)) ms")
    print("Query p999:          \(String(format: "%.2f", results.latencyDistribution.p999Ms)) ms")
    print("Query stddev:        \(String(format: "%.3f", results.queryLatencyStdDevMs)) ms")
    print("Query QPS:           \(String(format: "%.2f", results.qps))")
    print("Recall@1:            \(String(format: "%.3f", results.recallAt1))")
    print("Recall@10:           \(String(format: "%.3f", results.recallAt10))")
    print("Recall@100:          \(String(format: "%.3f", results.recallAt100))")

    let indexResidentMB = Double(results.indexResidentBytesEstimate) / (1024 * 1024)
    print("Index resident:      \(String(format: "%.1f", indexResidentMB)) MB")
    print("Peak resident:       \(String(format: "%.1f", results.memoryAfterQueries.peakResidentMB)) MB")

    if let environment {
        print("Device:              \(environment.metalDeviceName ?? "CPU only")")
        if environment.thermalState != "nominal" {
            print("Thermal:             \(environment.thermalState) (warning)")
        } else {
            print("Thermal:             \(environment.thermalState)")
        }
    }

    print("Latency histogram:")
    print(results.latencyDistribution.renderASCIIHistogram(width: 30))
}

func printRunBanner(_ environment: EnvironmentProbe) {
    let memoryGiB = Double(environment.physicalMemoryBytes) / (1024.0 * 1024.0 * 1024.0)
    let deviceLine = environment.metalDeviceName ?? "(no Metal)"
    print("== MetalANNS Benchmarks ==")
    print("Device: \(deviceLine)  Cores: \(environment.activeCoreCount)  Mem: \(String(format: "%.1f", memoryGiB)) GiB")
    let buildSuffix = environment.osBuild.isEmpty ? "" : " (\(environment.osBuild))"
    print("OS:     \(environment.osVersion)\(buildSuffix)")
    print("Mode:   \(environment.buildConfiguration)  Thermal: \(environment.thermalState)  LowPower: \(environment.lowPowerModeEnabled)")
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

func makeCrossProductConfigs(
    from base: BenchmarkRunner.Config,
    degreeValues: [Int],
    efSearchValues: [Int],
    dataset: BenchmarkDataset
) -> [(label: String, config: BenchmarkRunner.Config)] {
    var combos: [(label: String, config: BenchmarkRunner.Config)] = []
    combos.reserveCapacity(degreeValues.count * efSearchValues.count)
    for degree in degreeValues {
        for efSearch in efSearchValues {
            combos.append(
                (
                    label: "degree=\(degree),efSearch=\(efSearch)",
                    config: BenchmarkRunner.Config(
                        vectorCount: dataset.trainVectors.count,
                        dim: dataset.dimension,
                        degree: degree,
                        queryCount: base.queryCount,
                        k: base.k,
                        efSearch: efSearch,
                        metric: base.metric
                    )
                )
            )
        }
    }
    return combos
}

func printUsage() {
    print("USAGE:")
    print("  MetalANNSBenchmarks [--dataset <path.annbin>]                                  # synthetic or loaded single run (default)")
    print("  MetalANNSBenchmarks --sweep [--dataset <path.annbin>]                          # sweep efSearch")
    print("  MetalANNSBenchmarks --dataset <path.annbin> --sweep [--sweep-efsearch <list>]  # dataset-aware sweep")
    print("  MetalANNSBenchmarks --dataset <path.annbin> --csv-out <path.csv>                # save CSV")
    print("  MetalANNSBenchmarks --ivfpq                                                    # ANS vs IVFPQ (synthetic if no dataset)")
    print("  MetalANNSBenchmarks --concurrency 8                                            # single run with N in-flight queries")
    print("  MetalANNSBenchmarks --concurrency-sweep 1,2,4,8,16                              # sweep concurrency levels")
    print("  MetalANNSBenchmarks --compare cpu,gpu,gpu-adc                                   # multi-backend compare (see note)")
    print("\nOPTIONS:")
    print("  --query-count <n>        override number of query vectors")
    print("  --seed <n>               deterministic seed for synthetic dataset")
    print("  --runs <n>               number of measured benchmark passes")
    print("  --warmup <n>             number of warmup passes")
    print("  --sweep-efsearch <list>  comma-separated ef values (default: 16,32,64,128,256)")
    print("  --sweep-degree <list>    comma-separated degree values; combined with --sweep-efsearch")
    print("                           emits the cross-product as 'degree=D,efSearch=E' rows")
    print("  --dataset <path.annbin>   use real dataset")
    print("  --csv-out <path.csv>      save CSV report")
    print("  --json-out <path.json>    save JSON report")
    print("  --histogram-out <path>    on a single run, write <path>.histogram.csv and <path>.cdf.csv")
    print("  --metric <cosine|l2|innerproduct|hamming>")
    print("  --degree <n>")
    print("  --efsearch <n>")
    print("  --k <n>")
    print("  --concurrency <n>           single concurrency level for normal runs (default 1)")
    print("  --concurrency-sweep <list>  comma-separated concurrency levels (e.g. 1,2,4,8,16); switches mode to concurrency sweep")
    print("  --compare <list>            comma-separated backend labels; emits one row per label (NOTE: _GraphIndex has no public backend selector — see warning at startup)")
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
        case concurrencySweep
        case compare
        case ivfpq
        case help
    }

    var mode: Mode = .single
    var shouldPrintUsage = false
    var datasetPath: String?
    var csvOutPath: String?
    var jsonOutPath: String?
    var queryCount: Int?
    var seed: Int? = 42
    var repeatRuns: Int = 1
    var warmupRuns: Int = 0
    var sweepEfSearchValues: [Int] = []
    var sweepDegreeValues: [Int] = []
    var histogramOutPath: String?
    var metric: Metric?
    var degree: Int?
    var efSearch: Int?
    var k: Int?
    var concurrency: Int = 1
    var concurrencySweepLevels: [Int] = []
    var compareBackendLabels: [String] = []

    var ivfpqSubspaces: Int = 8
    var ivfpqNumCentroids: Int = 256
    var ivfpqNumCoarseCentroids: Int = 256
    var ivfpqNprobe: Int = 8
    var ivfpqTrainingIterations: Int = 10
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
            options.sweepEfSearchValues = try parseIntList(value, flag: arg)

        case "--sweep-degree":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.sweepDegreeValues = try parseIntList(value, flag: arg)

        case "--histogram-out":
            options.histogramOutPath = try nextValue(for: arg, args: args, index: &index)

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

        case "--concurrency":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.concurrency = try parsePositiveInt(arg, value)

        case "--concurrency-sweep":
            let value = try nextValue(for: arg, args: args, index: &index)
            options.concurrencySweepLevels = try parseIntList(value)
            options.mode = .concurrencySweep

        case "--compare":
            let value = try nextValue(for: arg, args: args, index: &index)
            let labels = value
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            guard !labels.isEmpty else {
                throw BenchmarkDatasetError.invalidDataset("Invalid value for --compare: \(value)")
            }
            options.compareBackendLabels = labels
            options.mode = .compare

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

func parseIntList(_ value: String, flag: String = "--sweep-efsearch") throws -> [Int] {
    let values = value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .compactMap { Int($0) }
        .filter { $0 > 0 }

    if values.isEmpty {
        throw BenchmarkDatasetError.invalidDataset("Invalid list for \(flag): \(value)")
    }

    return values
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
    let options = try parseOptions(from: args)

    if options.shouldPrintUsage {
        printUsage()
        exit(0)
    }

    let environment = EnvironmentProbe.capture()
    printRunBanner(environment)

    switch options.mode {
    case .help:
        printUsage()
        exit(0)

    case .single:
        var datasetLabel = "synthetic"
        let dataset: BenchmarkDataset
        if let datasetPath = options.datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
            datasetLabel = datasetPath
        } else {
            let queryCount = options.queryCount ?? 100
            dataset = BenchmarkDataset.synthetic(
                trainCount: 1000,
                testCount: queryCount,
                dimension: 128,
                k: max(100, options.k ?? 10),
                metric: options.metric ?? .cosine,
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

        let results: BenchmarkRunner.Results
        if options.concurrency > 1 {
            results = try await BenchmarkRunner.runConcurrent(
                config: baseConfig,
                dataset: dataset,
                concurrency: options.concurrency,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns
            )
        } else {
            results = try await BenchmarkRunner.run(
                config: baseConfig,
                dataset: dataset,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns
            )
        }

        printResults(results, environment: environment)
        print("Dataset: \(datasetLabel)")
        if options.concurrency > 1 {
            print("Concurrency:         \(options.concurrency)")
        }
        print("First-query (cold):  \(String(format: "%.2f", results.firstQueryLatencyMs)) ms")
        print("Warm steady mean:    \(String(format: "%.3f", results.warmSteadyMeanMs)) ms")

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
                seed: options.seed
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
        if let histogramOutPath = options.histogramOutPath {
            let histogramURL = URL(fileURLWithPath: "\(histogramOutPath).histogram.csv")
            let cdfURL = URL(fileURLWithPath: "\(histogramOutPath).cdf.csv")
            let parentURL = histogramURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(
                at: parentURL,
                withIntermediateDirectories: true
            )
            try results.latencyDistribution.histogramCSV().write(to: histogramURL, atomically: true, encoding: .utf8)
            try results.latencyDistribution.cdfCSV().write(to: cdfURL, atomically: true, encoding: .utf8)
            print("Saved histogram: \(histogramURL.path)")
            print("Saved CDF: \(cdfURL.path)")
        }

    case .sweep:
        let dataset: BenchmarkDataset
        let datasetLabel: String

        if let datasetPath = options.datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
            datasetLabel = datasetPath
        } else {
            let base = try loadOrSyntheticDataset(
                path: nil,
                baseConfig: BenchmarkRunner.Config(
                    vectorCount: 1000,
                    dim: 128,
                    queryCount: options.queryCount ?? 100,
                    k: options.k ?? 10,
                    efSearch: options.efSearch ?? 64,
                    metric: options.metric ?? .cosine
                ),
                metric: options.metric ?? .cosine,
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

        let efValues = options.sweepEfSearchValues.isEmpty ? defaultSweepValues : options.sweepEfSearchValues
        let configs: [(label: String, config: BenchmarkRunner.Config)]
        if !options.sweepDegreeValues.isEmpty {
            configs = makeCrossProductConfigs(
                from: base,
                degreeValues: options.sweepDegreeValues,
                efSearchValues: efValues,
                dataset: dataset
            )
        } else {
            configs = makeConfigsForSweep(from: base, values: efValues, dataset: dataset)
        }

        let report = try await BenchmarkRunner.sweep(
            configs: configs,
            dataset: dataset,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns
        )
        print(report.renderTable())
        print("Pareto frontier points: \(report.paretoFrontier().count)")
        print("")
        print(report.renderParetoChart(width: 60, height: 16))

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
                seed: options.seed
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

    case .concurrencySweep:
        let dataset: BenchmarkDataset
        let datasetLabel: String

        if let datasetPath = options.datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
            datasetLabel = datasetPath
        } else {
            let base = try loadOrSyntheticDataset(
                path: nil,
                baseConfig: BenchmarkRunner.Config(
                    vectorCount: 1000,
                    dim: 128,
                    queryCount: options.queryCount ?? 100,
                    k: options.k ?? 10,
                    efSearch: options.efSearch ?? 64,
                    metric: options.metric ?? .cosine
                ),
                metric: options.metric ?? .cosine,
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

        let levels = options.concurrencySweepLevels.isEmpty ? [1, 2, 4, 8, 16] : options.concurrencySweepLevels
        let report = try await BenchmarkRunner.concurrencySweep(
            config: base,
            dataset: dataset,
            concurrencyLevels: levels,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns
        )
        print(report.renderTable())

        let metadataReport = BenchmarkReport(
            rows: report.rows,
            datasetLabel: report.datasetLabel,
            metadata: benchmarkMetadata(
                mode: "concurrencySweep",
                config: base,
                datasetLabel: datasetLabel,
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed
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

    case .compare:
        // _GraphIndex does not currently expose a public knob for selecting between
        // its CPU / GPU / GPU-ADC search backends — the choice is made internally
        // based on workload size, metric, and EF parameters. We emit one row per
        // requested label so users can still measure variance, but they should
        // interpret the rows accordingly.
        fputs(
            "warning: --compare requested but _GraphIndex has no public backend selector. " +
            "All rows below run the same auto-selected backend; per-label values reflect " +
            "run-to-run variance, not differences between distinct backends.\n",
            stderr
        )

        let dataset: BenchmarkDataset
        let datasetLabel: String

        if let datasetPath = options.datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
            datasetLabel = datasetPath
        } else {
            let base = try loadOrSyntheticDataset(
                path: nil,
                baseConfig: BenchmarkRunner.Config(
                    vectorCount: 1000,
                    dim: 128,
                    queryCount: options.queryCount ?? 100,
                    k: options.k ?? 10,
                    efSearch: options.efSearch ?? 64,
                    metric: options.metric ?? .cosine
                ),
                metric: options.metric ?? .cosine,
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

        let report = try await BenchmarkRunner.compareBackends(
            config: base,
            dataset: dataset,
            backendLabels: options.compareBackendLabels,
            repeatRuns: options.repeatRuns,
            warmupRuns: options.warmupRuns,
            concurrency: options.concurrency
        )
        print(report.renderTable())

        let metadataReport = BenchmarkReport(
            rows: report.rows,
            datasetLabel: report.datasetLabel,
            metadata: benchmarkMetadata(
                mode: "compare",
                config: base,
                datasetLabel: datasetLabel,
                csvOut: options.csvOutPath,
                repeatRuns: options.repeatRuns,
                warmupRuns: options.warmupRuns,
                seed: options.seed
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
        let datasetSource = try loadOrSyntheticDataset(
            path: options.datasetPath,
            baseConfig: BenchmarkRunner.Config(
                vectorCount: 1000,
                dim: 128,
                queryCount: options.queryCount ?? 100,
                k: options.k ?? 10,
                efSearch: options.efSearch ?? 64,
                metric: options.metric ?? .cosine
            ),
            metric: options.metric ?? .l2,
            seed: options.seed ?? 42
        )
        let dataset = datasetSource.dataset
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
                seed: options.seed
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
    }
} catch {
    fputs("Benchmark failed: \(error)\n", stderr)
    printUsage()
    exit(1)
}
