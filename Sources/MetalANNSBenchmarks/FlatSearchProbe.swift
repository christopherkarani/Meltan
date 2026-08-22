import Foundation
import Metal
import MetalANNSCore

/// Component-level probe of the fused exact-search path: times production
/// entry points against a raw VectorBuffer corpus so kernel/dispatch/selection
/// costs can be attributed without graph-index overhead.
enum FlatSearchProbe {
    struct Sample {
        let vectorCount: Int
        let dim: Int
        let singleP50Us: Double
        let singleP10Us: Double
        let batchPerQueryP50Us: Double
        let corpusMB: Double
    }

    static func run(vectorCounts: [Int], dims: [Int], queriesPerBatch: Int = 64) async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        let context = try MetalContext()
        print("== FlatSearchProbe (\(device.name)) ==")

        func percentile(_ xs: [Double], _ p: Double) -> Double {
            let sorted = xs.sorted()
            return sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))]
        }

        for n in vectorCounts {
            for dim in dims {
                let buffer = try VectorBuffer(capacity: n, dim: dim, device: device)
                var state: UInt64 = 0x9E37_79B9_7F4A_7C15
                func next() -> Float {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2.0 - 1.0
                }
                var offset = 0
                while offset < n {
                    let take = min(4096, n - offset)
                    try buffer.batchInsert(
                        vectors: (0..<take).map { _ in (0..<dim).map { _ in next() } },
                        startingAt: offset)
                    offset += take
                }
                buffer.setCount(n)
                let query = (0..<dim).map { _ in next() }

                var samples: [Double] = []
                samples.reserveCapacity(60)
                // Sustained burn-in: Apple GPU clocks ramp under load, so
                // short warmups sample the ramp, not steady state. The
                // end-to-end harness reaches steady state via multi-second
                // graph construction; mirror that here before sampling.
                for _ in 0..<400 {
                    _ = try await awaitSearch(context: context, query: query, buffer: buffer, dim: dim)
                }
                for _ in 0..<60 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = try await awaitSearch(context: context, query: query, buffer: buffer, dim: dim)
                    samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3)
                }

                let batchQueries = (0..<queriesPerBatch).map { _ in (0..<dim).map { _ in next() } }
                var batchSamples: [Double] = []
                for _ in 0..<30 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = try await awaitBatchSearch(
                        context: context, queries: batchQueries, buffer: buffer, dim: dim)
                    batchSamples.append(
                        Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
                            / Double(batchQueries.count))
                }

                let row = Sample(
                    vectorCount: n,
                    dim: dim,
                    singleP50Us: percentile(samples, 0.5),
                    singleP10Us: percentile(samples, 0.1),
                    batchPerQueryP50Us: percentile(batchSamples, 0.5),
                    corpusMB: Double(n * dim * MemoryLayout<Float>.size) / 1_048_576)
                print(
                    String(
                        format:
                            "n=%6d dim=%3d | corpus %6.1f MB | single p10=%5.0fus p50=%5.0fus | batch/%dq per-query p50=%5.0fus",
                        row.vectorCount, row.dim, row.corpusMB,
                        row.singleP10Us, row.singleP50Us, queriesPerBatch, row.batchPerQueryP50Us))
            }
        }
    }

    private static func awaitSearch(
        context: MetalContext, query: [Float], buffer: VectorBuffer, dim: Int
    ) async throws -> [SearchResult] {
        try await FlatGPUSearch.search(
            context: context, query: query, vectors: buffer, k: 24, metric: .cosine)
    }

    private static func awaitBatchSearch(
        context: MetalContext, queries: [[Float]], buffer: VectorBuffer, dim: Int
    ) async throws -> [[SearchResult]] {
        try await FlatGPUSearch.batchSearch(
            context: context, queries: queries, vectors: buffer, k: 24, metric: .cosine)
    }
}
