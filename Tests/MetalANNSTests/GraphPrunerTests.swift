import Foundation
import Metal
import Testing
@testable import MetalANNSCore

@Suite("Graph Pruner Tests")
struct GraphPrunerTests {
    private func randomVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { _ in
            seededVector(dim: dim)
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
            var paddedIDs = ids + Array(
                repeating: UInt32.max,
                count: max(0, degree - ids.count)
            )
            var paddedDists = dists + Array(
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

    private func recall(
        context: MetalContext,
        queries: [[Float]],
        vectors: [[Float]],
        vectorBuffer: VectorBuffer,
        graphBuffer: GraphBuffer,
        entryPoint: UInt32,
        k: Int,
        ef: Int
    ) async throws -> Float {
        var total: Float = 0
        for query in queries {
            let exact = exactTopK(query: query, vectors: vectors, k: k)
            let approx = try await FullGPUSearch.search(
                context: context,
                query: query,
                vectors: vectorBuffer,
                graph: graphBuffer,
                entryPoint: Int(entryPoint),
                k: k,
                ef: ef,
                metric: .cosine
            )
            let approxSet = Set(approx.map(\.internalID))
            total += Float(exact.intersection(approxSet).count) / Float(k)
        }
        return total / Float(queries.count)
    }

    @Test("Pruning reduces redundant edges without over-pruning")
    func pruningReducesRedundancy() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 200
        let dim = 16
        let degree = 8
        let context = try MetalContext()
        let vectors = randomVectors(count: nodeCount, dim: dim)
        let (vectorBuffer, graphBuffer, _) = try await buildBuffers(
            context: context,
            vectors: vectors,
            degree: degree,
            metric: .cosine
        )

        let edgesBefore = (0..<nodeCount).reduce(into: 0) { total, node in
            total += graphBuffer.neighborIDs(of: node).filter { $0 != UInt32.max }.count
        }

        try GraphPruner.prune(
            graph: graphBuffer,
            vectors: vectorBuffer,
            nodeCount: nodeCount,
            metric: .cosine
        )

        let edgesAfter = (0..<nodeCount).reduce(into: 0) { total, node in
            total += graphBuffer.neighborIDs(of: node).filter { $0 != UInt32.max }.count
        }
        let averageAfter = Float(edgesAfter) / Float(nodeCount)
        print("GraphPrunerTests average neighbors after pruning: \(averageAfter)")

        #expect(edgesAfter <= edgesBefore)
        #expect(edgesAfter > 0)
        #expect(averageAfter > Float(degree) / 2.0)
    }

    @Test("Pruning preserves recall within 2%")
    func pruningMaintainsRecall() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 300
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
        let queries = randomVectors(count: queryCount, dim: dim)

        let baselineRecall = try await recall(
            context: context,
            queries: queries,
            vectors: vectors,
            vectorBuffer: vectorBuffer,
            graphBuffer: graphBuffer,
            entryPoint: entryPoint,
            k: k,
            ef: ef
        )

        try GraphPruner.prune(
            graph: graphBuffer,
            vectors: vectorBuffer,
            nodeCount: nodeCount,
            metric: .cosine
        )

        let prunedRecall = try await recall(
            context: context,
            queries: queries,
            vectors: vectors,
            vectorBuffer: vectorBuffer,
            graphBuffer: graphBuffer,
            entryPoint: entryPoint,
            k: k,
            ef: ef
        )
        print("GraphPrunerTests recall baseline: \(baselineRecall), pruned: \(prunedRecall)")

        #expect(prunedRecall > baselineRecall - 0.02)
    }
}
