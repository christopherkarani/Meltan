import Testing

@testable import MetalANNSCore

@Suite("GraphBuffer Tests")
struct GraphBufferTests {
    @Test("Set and read neighbors for node 0")
    func setAndReadNeighbors() throws {
        let graph = try GraphBuffer(capacity: 10, degree: 4)
        let neighborIDs: [UInt32] = [3, 7, 1, 9]
        let neighborDistances: [Float] = [0.1, 0.3, 0.05, 0.8]

        try graph.setNeighbors(of: 0, ids: neighborIDs, distances: neighborDistances)

        #expect(graph.neighborIDs(of: 0) == neighborIDs)
        #expect(graph.neighborDistances(of: 0) == neighborDistances)
    }

    @Test("Nodes are independent")
    func nodeIndependence() throws {
        let graph = try GraphBuffer(capacity: 10, degree: 2)

        try graph.setNeighbors(of: 0, ids: [1, 2], distances: [0.1, 0.2])
        try graph.setNeighbors(of: 1, ids: [5, 6], distances: [0.5, 0.6])

        #expect(graph.neighborIDs(of: 0) == [1, 2])
        #expect(graph.neighborDistances(of: 0) == [0.1, 0.2])
        #expect(graph.neighborIDs(of: 1) == [5, 6])
        #expect(graph.neighborDistances(of: 1) == [0.5, 0.6])
    }

    @Test("Capacity and degree are correct")
    func capacityAndDegree() throws {
        let graph = try GraphBuffer(capacity: 100, degree: 32)

        #expect(graph.capacity == 100)
        #expect(graph.degree == 32)
    }
}
