import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Metal Search Tests")
struct MetalSearchTests {
    private func entryPoint(for graph: GraphBuffer, nodeCount: Int) -> Int {
        var bestNode = 0
        var bestMean = Float.greatestFiniteMagnitude

        for node in 0..<nodeCount {
            let distances = graph.neighborDistances(of: node)
            guard !distances.isEmpty else { continue }
            let mean = distances.reduce(Float(0), +) / Float(distances.count)
            if mean < bestMean {
                bestMean = mean
                bestNode = node
            }
        }

        return bestNode
    }

    @Test("GPU beam search returns k results")
    func gpuSearchReturnsK() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 100
        let dim = 16
        let degree = 8
        let k = 5
        let ef = 32
        let vectors = seededVectors(count: nodeCount, dim: dim)

        let context = try MetalContext()
        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: context.device)
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)
        try await NNDescentGPU.build(
            context: context,
            vectors: vectorBuffer,
            graph: graph,
            nodeCount: nodeCount,
            metric: .cosine,
            maxIterations: 15
        )

        let query = seededVector(dim: dim)
        let results = try await SearchGPU.search(
            context: context,
            query: query,
            vectors: vectorBuffer,
            graph: graph,
            entryPoint: entryPoint(for: graph, nodeCount: nodeCount),
            k: k,
            ef: ef,
            metric: .cosine
        )

        #expect(results.count == k)
        for index in 1..<results.count {
            #expect(results[index].score >= results[index - 1].score)
        }
    }

    @Test("GPU search recall > 0.85 on 500 vectors")
    func gpuSearchRecall() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 500
        let dim = 32
        let degree = 16
        let k = 10
        let ef = 64
        let queryCount = 10
        let vectors = seededVectors(count: nodeCount, dim: dim)

        let context = try MetalContext()
        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: context.device)
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)
        try await NNDescentGPU.build(
            context: context,
            vectors: vectorBuffer,
            graph: graph,
            nodeCount: nodeCount,
            metric: .cosine,
            maxIterations: 15
        )

        var totalRecall: Float = 0

        for _ in 0..<queryCount {
            let query = seededVector(dim: dim)
            let results = try await SearchGPU.search(
                context: context,
                query: query,
                vectors: vectorBuffer,
                graph: graph,
                entryPoint: entryPoint(for: graph, nodeCount: nodeCount),
                k: k,
                ef: ef,
                metric: .cosine
            )

            let exactDistances = vectors.map { SIMDDistance.distance($0, query, metric: .cosine) }

            let exactTopK = Set(
                exactDistances.enumerated()
                    .sorted { $0.element < $1.element }
                    .prefix(k)
                    .map { UInt32($0.offset) }
            )
            let approxTopK = Set(results.map(\.internalID))
            let overlap = exactTopK.intersection(approxTopK).count
            totalRecall += Float(overlap) / Float(k)
        }

        let averageRecall = totalRecall / Float(queryCount)
        #expect(averageRecall > 0.85, "Average recall \\(averageRecall) below 0.85")
    }
}
