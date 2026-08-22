import Foundation
import Metal
import Testing

@testable import MetalANNSCore

@Suite("GPU-CPU Search Parity")
struct GPUCPUParityTests {
    private func makeGPUContextOrSkip() -> MetalContext? {
        #if targetEnvironment(simulator)
            print("Skipping GPUCPUParityTests on simulator")
            return nil
        #else
            guard MTLCreateSystemDefaultDevice() != nil else {
                print("Skipping GPUCPUParityTests: no Metal device available")
                return nil
            }
            do {
                return try MetalContext()
            } catch {
                print("Skipping GPUCPUParityTests: MetalContext unavailable (\(error))")
                return nil
            }
        #endif
    }

    private func buildGraph(
        vectors: [[Float]],
        degree: Int,
        maxIterations: Int,
        context: MetalContext
    ) async throws -> (
        vectorBuffer: VectorBuffer, graphBuffer: GraphBuffer, cpuGraph: [[(UInt32, Float)]], entryPoint: Int
    ) {
        let nodeCount = vectors.count
        let dim = vectors[0].count
        let device = context.device

        let (cpuGraph, cpuEntry) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: maxIterations
        )

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graphBuffer = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        for node in 0..<nodeCount {
            let neighbors = cpuGraph[node]
            let ids = neighbors.map(\.0) + Array(repeating: UInt32.max, count: max(0, degree - neighbors.count))
            let dists =
                neighbors.map(\.1)
                + Array(repeating: Float.greatestFiniteMagnitude, count: max(0, degree - neighbors.count))
            try graphBuffer.setNeighbors(
                of: node,
                ids: Array(ids.prefix(degree)),
                distances: Array(dists.prefix(degree))
            )
        }
        graphBuffer.setCount(nodeCount)

        return (vectorBuffer, graphBuffer, cpuGraph, Int(cpuEntry))
    }

    @Test(arguments: [
        (nodeCount: 100, dim: 32, degree: 8, k: 5, ef: 32, maxIter: 10),
        (nodeCount: 500, dim: 64, degree: 16, k: 10, ef: 64, maxIter: 10),
        (nodeCount: 2000, dim: 128, degree: 32, k: 20, ef: 128, maxIter: 5),
        (nodeCount: 8000, dim: 384, degree: 32, k: 10, ef: 64, maxIter: 3),
    ])
    func gpuAndCPUSearchAgreeOnSameGraph(
        nodeCount: Int,
        dim: Int,
        degree: Int,
        k: Int,
        ef: Int,
        maxIter: Int
    ) async throws {
        guard let context = makeGPUContextOrSkip() else {
            return
        }

        var rng = SeededGenerator(state: UInt64(nodeCount) &* UInt64(dim) &+ 7)
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let (vectorBuffer, graphBuffer, cpuGraph, entryPoint) = try await buildGraph(
            vectors: vectors,
            degree: degree,
            maxIterations: maxIter,
            context: context
        )

        let stride = max(1, nodeCount / 5)
        var totalOverlap: Float = 0
        var queryCount = 0

        for qi in 0..<5 {
            let queryIndex = qi * stride
            guard queryIndex < nodeCount else {
                continue
            }

            let query = vectors[queryIndex]
            let gpuResults = try await FullGPUSearch.search(
                context: context,
                query: query,
                vectors: vectorBuffer,
                graph: graphBuffer,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: .cosine
            )

            let cpuResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: cpuGraph,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: .cosine
            )

            guard !gpuResults.isEmpty, !cpuResults.isEmpty else {
                continue
            }

            let gpuIDs = Set(gpuResults.prefix(k).map(\.internalID))
            let cpuIDs = Set(cpuResults.prefix(k).map(\.internalID))
            let denominator = Float(min(k, min(gpuResults.count, cpuResults.count)))
            let overlap = Float(gpuIDs.intersection(cpuIDs).count) / denominator

            totalOverlap += overlap
            queryCount += 1
        }

        guard queryCount > 0 else {
            Issue.record("No queries produced results at n=\(nodeCount) dim=\(dim)")
            return
        }

        let avgOverlap = totalOverlap / Float(queryCount)
        let overlapText = String(format: "%.2f", avgOverlap)
        #expect(
            avgOverlap >= 0.6,
            "GPU-vs-CPU avg overlap \(overlapText) < 0.60 at n=\(nodeCount) dim=\(dim) degree=\(degree) k=\(k) ef=\(ef)"
        )
    }

    @Test("GPU search deterministic for same query")
    func gpuSearchIsDeterministic() async throws {
        guard let context = makeGPUContextOrSkip() else {
            return
        }
        var rng = SeededGenerator(state: 1234)

        let nodeCount = 300
        let dim = 64
        let degree = 16
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let (vectorBuffer, graphBuffer, _, entryPoint) = try await buildGraph(
            vectors: vectors,
            degree: degree,
            maxIterations: 5,
            context: context
        )

        let query = vectors[10]
        let run1 = try await FullGPUSearch.search(
            context: context,
            query: query,
            vectors: vectorBuffer,
            graph: graphBuffer,
            entryPoint: entryPoint,
            k: 10,
            ef: 32,
            metric: .cosine
        )
        let run2 = try await FullGPUSearch.search(
            context: context,
            query: query,
            vectors: vectorBuffer,
            graph: graphBuffer,
            entryPoint: entryPoint,
            k: 10,
            ef: 32,
            metric: .cosine
        )

        #expect(run1.map(\.internalID) == run2.map(\.internalID))
    }
}
