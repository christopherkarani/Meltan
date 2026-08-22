import Metal
import Testing

@testable import MetalANNSCore

@Suite("Full GPU Search Tests")
struct FullGPUSearchTests {
    private func randomVectors(count: Int, dim: Int, seed: UInt64 = 42) -> [[Float]] {
        (0..<count).map { i in
            seededVector(dim: dim, seed: seed + UInt64(i))
        }
    }

    private func buildBuffers(
        context: MetalContext,
        vectors: [[Float]],
        degree: Int,
        metric: Metric
    ) async throws -> (VectorBuffer, GraphBuffer, UInt32) {
        let nodeCount = vectors.count
        let dim = vectors[0].count

        let (cpuGraph, cpuEntry) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: metric,
            maxIterations: 10
        )

        let vectorBuffer = try VectorBuffer(
            capacity: nodeCount,
            dim: dim,
            device: context.device
        )
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graphBuffer = try GraphBuffer(
            capacity: nodeCount,
            degree: degree,
            device: context.device
        )

        for node in 0..<nodeCount {
            let ids = cpuGraph[node].map(\.0)
            let dists = cpuGraph[node].map(\.1)
            var paddedIDs =
                ids
                + Array(
                    repeating: UInt32.max,
                    count: max(0, degree - ids.count)
                )
            var paddedDists =
                dists
                + Array(
                    repeating: Float.greatestFiniteMagnitude,
                    count: max(0, degree - dists.count)
                )
            paddedIDs = Array(paddedIDs.prefix(degree))
            paddedDists = Array(paddedDists.prefix(degree))
            try graphBuffer.setNeighbors(of: node, ids: paddedIDs, distances: paddedDists)
        }
        graphBuffer.setCount(nodeCount)

        return (vectorBuffer, graphBuffer, cpuEntry)
    }

    private func exactTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int
    ) -> Set<UInt32> {
        Set(
            vectors.enumerated()
                .map { (idx, vector) in (UInt32(idx), SIMDDistance.cosine(query, vector)) }
                .sorted { $0.1 < $1.1 }
                .prefix(k)
                .map(\.0)
        )
    }

    @Test("Full GPU search returns k sorted results")
    func gpuSearchReturnsK() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 200
        let dim = 16
        let degree = 8
        let context = try MetalContext()
        let vectors = randomVectors(count: nodeCount, dim: dim)
        let (vectorBuffer, graphBuffer, entryPoint) = try await buildBuffers(
            context: context,
            vectors: vectors,
            degree: degree,
            metric: .cosine
        )

        let query = seededVector(dim: dim, seed: 999)
        let results = try await FullGPUSearch.search(
            context: context,
            query: query,
            vectors: vectorBuffer,
            graph: graphBuffer,
            entryPoint: Int(entryPoint),
            k: 5,
            ef: 32,
            metric: .cosine
        )

        #expect(results.count == 5)
        for index in 1..<results.count {
            #expect(results[index].score >= results[index - 1].score)
        }
    }

    @Test("Full GPU recall matches hybrid recall within tolerance")
    func gpuSearchRecallMatchesHybrid() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 500
        let dim = 32
        let degree = 16
        let k = 10
        let ef = 64
        let queryCount = 20

        let context = try MetalContext()
        let vectors = randomVectors(count: nodeCount, dim: dim)
        let (vectorBuffer, graphBuffer, entryPoint) = try await buildBuffers(
            context: context,
            vectors: vectors,
            degree: degree,
            metric: .cosine
        )

        var fullRecallTotal: Float = 0
        var hybridRecallTotal: Float = 0

        for i in 0..<queryCount {
            let query = seededVector(dim: dim, seed: 1000 + UInt64(i))
            let exact = exactTopK(query: query, vectors: vectors, k: k)

            let fullResults = try await FullGPUSearch.search(
                context: context,
                query: query,
                vectors: vectorBuffer,
                graph: graphBuffer,
                entryPoint: Int(entryPoint),
                k: k,
                ef: ef,
                metric: .cosine
            )
            let hybridResults = try await SearchGPU.search(
                context: context,
                query: query,
                vectors: vectorBuffer,
                graph: graphBuffer,
                entryPoint: Int(entryPoint),
                k: k,
                ef: ef,
                metric: .cosine
            )

            let fullSet = Set(fullResults.map(\.internalID))
            let hybridSet = Set(hybridResults.map(\.internalID))

            fullRecallTotal += Float(exact.intersection(fullSet).count) / Float(k)
            hybridRecallTotal += Float(exact.intersection(hybridSet).count) / Float(k)
        }

        let fullRecall = fullRecallTotal / Float(queryCount)
        let hybridRecall = hybridRecallTotal / Float(queryCount)

        #expect(fullRecall > hybridRecall - 0.05)
        #expect(fullRecall > 0.80)
    }

    @Test("GPU search works above 4096-node old limit")
    func searchAbove4096NodesReturnsResults() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }

        let nodeCount = 5000
        let dim = 32
        let degree = 16
        let context = try MetalContext()
        let vectors = randomVectors(count: nodeCount, dim: dim)
        let (vectorBuffer, graphBuffer, entryPoint) = try await buildBuffers(
            context: context,
            vectors: vectors,
            degree: degree,
            metric: .cosine
        )

        let results = try await FullGPUSearch.search(
            context: context,
            query: vectors[0],
            vectors: vectorBuffer,
            graph: graphBuffer,
            entryPoint: Int(entryPoint),
            k: 10,
            ef: 64,
            metric: .cosine
        )

        #expect(results.count == 10, "GPU search at 5000 nodes should return 10 results")
        #expect(results[0].score < 0.05, "Top result should be near-exact match at large scale")
    }

    @Test("GPU search recall matches CPU at small scale")
    func gpuSearchMatchesCPUAtSmallScale() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }

        let nodeCount = 200
        let dim = 32
        let degree = 16
        let k = 10
        let ef = 64
        let context = try MetalContext()
        let vectors = randomVectors(count: nodeCount, dim: dim)
        let (vectorBuffer, graphBuffer, entryPoint) = try await buildBuffers(
            context: context,
            vectors: vectors,
            degree: degree,
            metric: .cosine
        )

        let query = vectors[5]
        let gpuResults = try await FullGPUSearch.search(
            context: context,
            query: query,
            vectors: vectorBuffer,
            graph: graphBuffer,
            entryPoint: Int(entryPoint),
            k: k,
            ef: ef,
            metric: .cosine
        )

        let exactTopK = Set(
            vectors.enumerated()
                .map { (idx, vector) in (UInt32(idx), SIMDDistance.cosine(query, vector)) }
                .sorted { $0.1 < $1.1 }
                .prefix(k)
                .map(\.0)
        )
        let gpuIDs = Set(gpuResults.map(\.internalID))
        let recall = Float(exactTopK.intersection(gpuIDs).count) / Float(k)

        #expect(recall >= 0.70, "GPU-vs-brute-force recall \(recall) is below threshold 0.70")
    }
}
