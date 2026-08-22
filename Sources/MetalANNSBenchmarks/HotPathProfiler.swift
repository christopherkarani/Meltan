import Accelerate
import Foundation
import Metal
import MetalANNS
import MetalANNSCore

/// Component-level hot-path profiler for the fused exact-search stack.
///
/// Decomposes warm single-query latency into its constituent costs:
/// fixed Metal dispatch tax, raw BLAS scan bandwidth, host-tier full path
/// (norms + top-K heap), GPU-tier round trip, batched amortization, and the
/// `_GraphIndex.search` actor/mapping overhead layered on top.
enum HotPathProfiler {
    struct Sample {
        let label: String
        let p50Us: Double
        let p95Us: Double
        let meanUs: Double
        let iterations: Int
    }

    static func run(dim: Int, k: Int, metric: Metric, sizes: [Int]) async throws {
        setvbuf(stdout, nil, _IOLBF, 8192)
        let effectiveSizes: [Int]
        if let override = ProcessInfo.processInfo.environment["PROFILE_SIZES"] {
            effectiveSizes =
                override
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        } else {
            effectiveSizes = sizes
        }
        print("== Hot-path profiler (dim=\(dim), k=\(k), metric=\(metric.rawValue)) ==")
        print("sizes: \(effectiveSizes)")
        print("creating MetalContext...")
        let context = try? MetalContext()
        if context == nil {
            print("No Metal device; GPU tiers will be skipped")
        }
        print("context ready")

        // Fixed dispatch tax: trivial kernel encode + commit + completion wait.
        if let context {
            let tax = await measureDispatchTax(context: context)
            printSample(tax)
        }

        for n in effectiveSizes {
            print("\n-- corpus n=\(n) --")
            let corpus = try makeCorpus(count: n, dim: dim)
            let queries = makeQueries(count: 64, dim: dim)
            let singleQuery = queries[0]

            // Raw BLAS scan floor.
            if let raw = measureRawScan(corpus: corpus, query: singleQuery, iterations: 60) {
                printSample(
                    Sample(label: "raw sgemv scan", p50Us: raw.p50, p95Us: raw.p95, meanUs: raw.mean, iterations: 60))
            }

            // Host tier full path (warm norm cache).
            var hostSamples: [Double] = []
            for _ in 0..<10 {
                _ = FlatGPUSearch.hostSearch(query: singleQuery, vectors: corpus, k: k, metric: metric)
            }
            for _ in 0..<120 {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = FlatGPUSearch.hostSearch(query: singleQuery, vectors: corpus, k: k, metric: metric)
                hostSamples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1000.0)
            }
            printSample(
                Sample(
                    label: "host tier (hostSearch)",
                    p50Us: percentile(hostSamples, 0.50),
                    p95Us: percentile(hostSamples, 0.95),
                    meanUs: hostSamples.reduce(0, +) / Double(hostSamples.count),
                    iterations: hostSamples.count
                )
            )

            guard let context else { continue }

            // GPU tier single-query round trip.
            if FlatGPUSearch.isEligible(vectors: corpus, metric: metric, k: k, maxVectorCount: .max) {
                var gpuWarmup = false
                for _ in 0..<5 {
                    gpuWarmup = true
                    _ = try await FlatGPUSearch.search(
                        context: context,
                        query: singleQuery,
                        vectors: corpus,
                        k: k,
                        metric: metric,
                        tierOverride: 0
                    )
                }
                _ = gpuWarmup
                var gpuSamples: [Double] = []
                gpuSamples.reserveCapacity(120)
                for _ in 0..<120 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = try await FlatGPUSearch.search(
                        context: context,
                        query: singleQuery,
                        vectors: corpus,
                        k: k,
                        metric: metric,
                        tierOverride: 0
                    )
                    gpuSamples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1000.0)
                }
                printSample(
                    Sample(
                        label: "GPU tier (search)",
                        p50Us: percentile(gpuSamples, 0.50),
                        p95Us: percentile(gpuSamples, 0.95),
                        meanUs: gpuSamples.reduce(0, +) / Double(gpuSamples.count),
                        iterations: gpuSamples.count
                    )
                )

                // Batched GPU amortization (64 queries in flight).
                var batchRuns = 0
                let batchStart = DispatchTime.now().uptimeNanoseconds
                _ = try await FlatGPUSearch.batchSearch(
                    context: context,
                    queries: queries,
                    vectors: corpus,
                    k: k,
                    metric: metric,
                    tierOverride: 0
                )
                batchRuns += 1
                _ = batchRuns
                let batchElapsedUs = Double(DispatchTime.now().uptimeNanoseconds - batchStart) / 1000.0
                let batchLabel = "GPU batchSearch x64".padding(toLength: 28, withPad: " ", startingAt: 0)
                print(
                    String(
                        format: "  %@ %8.1f us total for %d queries (%.1f us/query)",
                        batchLabel,
                        batchElapsedUs,
                        queries.count,
                        batchElapsedUs / Double(queries.count)
                    )
                )
            }

            // Full public path through the actor.
            let index = GraphIndex(
                configuration: IndexConfiguration(metric: metric, exactSearchMaxVectorCount: 1_000_000)
            )
            let ids = (0..<n).map { "v_\($0)" }
            let buildStart = DispatchTime.now().uptimeNanoseconds
            try await index.build(vectors: corpusVectors(corpus, count: n, dim: dim), ids: ids)
            let buildMs = Double(DispatchTime.now().uptimeNanoseconds - buildStart) / 1_000_000.0
            let buildLabel = "GraphIndex.build".padding(toLength: 28, withPad: " ", startingAt: 0)
            print(String(format: "  %@ %8.1f ms", buildLabel, buildMs))

            for _ in 0..<20 {
                _ = try await index.search(query: singleQuery, k: k)
            }
            var actorSamples: [Double] = []
            actorSamples.reserveCapacity(120)
            for _ in 0..<120 {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try await index.search(query: singleQuery, k: k)
                actorSamples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1000.0)
            }
            printSample(
                Sample(
                    label: "GraphIndex.search (actor)",
                    p50Us: percentile(actorSamples, 0.50),
                    p95Us: percentile(actorSamples, 0.95),
                    meanUs: actorSamples.reduce(0, +) / Double(actorSamples.count),
                    iterations: actorSamples.count
                )
            )

            if n >= 50_000 { break }  // keep profile runtime bounded
        }
    }

    // MARK: - Measurement helpers

    private static func measureDispatchTax(context: MetalContext) async -> Sample {
        print("measuring dispatch tax...")
        // Trivial workload: flat scan over 64 vectors = negligible GPU work;
        // measures encode + submit + completion-wait round trip.
        guard let tiny = try? makeCorpus(count: 64, dim: 16) else {
            return Sample(label: "dispatch tax", p50Us: -1, p95Us: -1, meanUs: -1, iterations: 0)
        }
        let query = [Float](repeating: 0.5, count: 16)
        var samples: [Double] = []
        samples.reserveCapacity(200)
        // Warm pipelines first.
        _ = try? await FlatGPUSearch.search(
            context: context, query: query, vectors: tiny, k: 8, metric: .l2, tierOverride: 0)
        print("pipeline warm")
        for _ in 0..<200 {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try? await FlatGPUSearch.search(
                context: context,
                query: query,
                vectors: tiny,
                k: 8,
                metric: .l2,
                tierOverride: 0
            )
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1000.0)
        }
        return Sample(
            label: "dispatch round trip (tiny)",
            p50Us: percentile(samples, 0.50),
            p95Us: percentile(samples, 0.95),
            meanUs: samples.reduce(0, +) / Double(samples.count),
            iterations: samples.count
        )
    }

    private static func measureRawScan(corpus: VectorBuffer, query: [Float], iterations: Int) -> (
        p50: Double, p95: Double, mean: Double
    )? {
        guard let corpusBase = corpus.floatPointer.baseAddress else { return nil }
        let n = corpus.count
        let dim = corpus.dim
        var output = [Float](repeating: 0, count: n)
        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            output.withUnsafeMutableBufferPointer { out in
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans, Int32(n), Int32(dim), 1.0,
                    corpusBase, Int32(dim), query, 1, 0.0, out.baseAddress!, 1
                )
            }
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1000.0)
        }
        return (
            percentile(samples, 0.50),
            percentile(samples, 0.95),
            samples.reduce(0, +) / Double(samples.count)
        )
    }

    private static func percentile(_ samples: [Double], _ q: Double) -> Double {
        guard !samples.isEmpty else { return -1 }
        let sorted = samples.sorted()
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * q).rounded())))
        return sorted[index]
    }

    private static func printSample(_ sample: Sample) {
        let label = sample.label.padding(toLength: 28, withPad: " ", startingAt: 0)
        print(
            String(
                format: "  %@ p50 %8.1f us   p95 %8.1f us   mean %8.1f us",
                label,
                sample.p50Us,
                sample.p95Us,
                sample.meanUs
            )
        )
    }

    // MARK: - Data helpers

    struct SeededRandom {
        var state: UInt64
        init(seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 0xD1B5_4A32_D192_ED03 }
        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1
        }
    }

    @discardableResult
    private static func makeCorpus(count: Int, dim: Int) throws -> VectorBuffer {
        let buffer = try VectorBuffer(capacity: count, dim: dim)
        var rng = SeededRandom(seed: 42)
        let chunkSize = 4096
        var start = 0
        while start < count {
            let end = min(start + chunkSize, count)
            var chunk = [[Float]]()
            chunk.reserveCapacity(end - start)
            for _ in start..<end {
                chunk.append((0..<dim).map { _ in rng.next() })
            }
            try buffer.batchInsert(vectors: chunk, startingAt: start)
            start = end
        }
        buffer.setCount(count)
        return buffer
    }

    private static func makeQueries(count: Int, dim: Int) -> [[Float]] {
        var rng = SeededRandom(seed: 999)
        return (0..<count).map { _ in (0..<dim).map { _ in rng.next() } }
    }

    private static func corpusVectors(_ corpus: VectorBuffer, count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { corpus.vector(at: $0) }
    }
}
