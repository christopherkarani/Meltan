import Testing

@testable import MetalANNSCore

@Suite("CPU NN-Descent Tests")
struct NNDescentCPUTests {
    @Test("Constructs graph with correct dimensions")
    func graphDimensions() async throws {
        let n = 50
        let dim = 8
        let degree = 4
        let vectors = seededVectors(count: n, dim: dim)

        let (graph, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: 10
        )

        #expect(graph.count == n)
        for neighbors in graph {
            #expect(neighbors.count == degree)
        }
    }

    @Test("No self-loops in constructed graph")
    func noSelfLoops() async throws {
        let n = 50
        let dim = 8
        let degree = 4
        let vectors = seededVectors(count: n, dim: dim)

        let (graph, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: 10
        )

        for (nodeID, neighbors) in graph.enumerated() {
            for (neighborID, _) in neighbors {
                #expect(neighborID != UInt32(nodeID), "Self-loop found at node \(nodeID)")
            }
        }
    }

    @Test("Recall > 0.85 for 50 nodes")
    func recallCheck() async throws {
        let n = 50
        let dim = 8
        let degree = 4
        let vectors = seededVectors(count: n, dim: dim)

        let (graph, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: 10
        )

        var totalRecall: Float = 0

        for i in 0..<n {
            let distances = vectors.map { SIMDDistance.distance($0, vectors[i], metric: .cosine) }

            let exactNeighbors = distances.enumerated()
                .filter { $0.offset != i }
                .sorted { $0.element < $1.element }
                .prefix(degree)
                .map { UInt32($0.offset) }
            let exact = Set(exactNeighbors)
            let approx = Set(graph[i].map { $0.0 })
            totalRecall += Float(exact.intersection(approx).count) / Float(degree)
        }

        let avgRecall = totalRecall / Float(n)
        #expect(avgRecall > 0.85, "Average recall \(avgRecall) is below 0.85")
    }
}
