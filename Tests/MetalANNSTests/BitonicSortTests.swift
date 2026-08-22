import Metal
import Testing

@testable import MetalANNSCore

@Suite("Bitonic Sort Tests")
struct BitonicSortTests {
    @Test("Sort neighbor lists by ascending distance")
    func sortNeighborLists() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let context = try MetalContext()
        let nodeCount = 100
        let degree = 32
        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)

        var rng = SeededGenerator(state: 42)
        for node in 0..<nodeCount {
            let ids = (0..<degree).map { _ in UInt32.random(in: 0..<UInt32(nodeCount), using: &rng) }
            let distances = (0..<degree).map { _ in Float.random(in: 0...10, using: &rng) }
            try graph.setNeighbors(of: node, ids: ids, distances: distances)
        }

        try await NNDescentGPU.sortNeighborLists(
            context: context,
            graph: graph,
            nodeCount: nodeCount
        )

        for node in 0..<nodeCount {
            let distances = graph.neighborDistances(of: node)
            for i in 1..<degree {
                #expect(
                    distances[i] >= distances[i - 1],
                    "Node \(node): distance[\(i)] (\(distances[i])) < distance[\(i - 1)] (\(distances[i - 1]))"
                )
            }
        }
    }
}
