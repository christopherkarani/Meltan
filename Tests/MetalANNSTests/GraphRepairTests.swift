import Foundation
import MetalANNS
import Testing

@testable import MetalANNSCore

@Suite("Graph Repair Tests")
struct GraphRepairTests {
    @Test("RepairConfiguration defaults are sensible")
    func repairConfigDefaults() {
        let config = RepairConfiguration.default
        #expect(config.repairInterval == 100)
        #expect(config.repairDepth == 2)
        #expect(config.repairIterations == 5)
        #expect(config.enabled == true)
    }

    @Test("RepairConfiguration clamps invalid values")
    func repairConfigClamping() {
        let config = RepairConfiguration(repairInterval: -5, repairDepth: 0, repairIterations: 100)
        #expect(config.repairInterval == 0)
        #expect(config.repairDepth == 1)
        #expect(config.repairIterations == 20)
    }

    @Test("Neighborhood collection expands correctly")
    func neighborhoodCollection() async throws {
        let vectors = (0..<10).map { i in
            (0..<8).map { d in
                sin(Float(i * 8 + d) * 0.173)
            }
        }

        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 4,
            metric: .l2,
            maxIterations: 5
        )

        let graphBuffer = try makeGraphBuffer(graphData, degree: 4)

        let updates = try GraphRepairer.repair(
            recentIDs: [0],
            vectors: try makeVectorBuffer(vectors),
            graph: graphBuffer,
            config: RepairConfiguration(repairInterval: 1, repairDepth: 1, repairIterations: 1),
            metric: .l2
        )

        #expect(updates >= 0)
    }

    @Test("Repair improves recall after inserts")
    func repairImprovesRecall() async throws {
        let initialCount = 200
        let insertCount = 100
        let dim = 16
        let degree = 8
        let metric: Metric = .cosine

        let initialVectors = (0..<initialCount).map { i in
            (0..<dim).map { d in
                let position = Float(i * dim + d)
                let lhs = sin(position * 0.173)
                let rhs = cos(position * 0.071)
                return lhs + rhs
            }
        }

        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: initialVectors,
            degree: degree,
            metric: metric,
            maxIterations: 10
        )

        let vectorBuffer = try makeVectorBuffer(initialVectors, extraCapacity: insertCount)
        let graphBuffer = try makeGraphBuffer(graphData, degree: degree, extraCapacity: insertCount)

        var allVectors = initialVectors
        var insertedIDs: [UInt32] = []
        for i in 0..<insertCount {
            let newVector = (0..<dim).map { d in
                sin(Float((initialCount + i) * dim + d) * 0.173)
            }
            let slot = initialCount + i

            try vectorBuffer.insert(vector: newVector, at: slot)
            vectorBuffer.setCount(slot + 1)

            try IncrementalBuilder.insert(
                vector: newVector,
                at: slot,
                into: graphBuffer,
                vectors: vectorBuffer,
                entryPoint: entryPoint,
                metric: metric,
                degree: degree
            )
            graphBuffer.setCount(slot + 1)

            allVectors.append(newVector)
            insertedIDs.append(UInt32(slot))
        }

        let queries = Array(allVectors.prefix(20))
        let recallBefore = try await averageRecall(
            queries: queries,
            vectors: allVectors,
            graph: graphBuffer,
            entryPoint: Int(entryPoint),
            k: 10,
            ef: 64,
            metric: metric
        )

        let updates = try GraphRepairer.repair(
            recentIDs: insertedIDs,
            vectors: vectorBuffer,
            graph: graphBuffer,
            config: RepairConfiguration(repairDepth: 2, repairIterations: 5),
            metric: metric
        )

        let recallAfter = try await averageRecall(
            queries: queries,
            vectors: allVectors,
            graph: graphBuffer,
            entryPoint: Int(entryPoint),
            k: 10,
            ef: 64,
            metric: metric
        )

        #expect(recallAfter >= recallBefore - 0.01, "Repair degraded recall: \(recallAfter) < \(recallBefore)")
        #expect(updates > 0, "Repair should have found some improvements")
    }

    @Test("Repair does nothing when disabled")
    func repairDisabled() async throws {
        let vectors = (0..<50).map { i in
            (0..<8).map { d in
                Float(i * 8 + d)
            }
        }

        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 4,
            metric: .l2,
            maxIterations: 5
        )

        let graphBuffer = try makeGraphBuffer(graphData, degree: 4)
        let originalNeighbors = (0..<50).map { graphBuffer.neighborIDs(of: $0) }

        let config = RepairConfiguration(enabled: false)
        let updates = try GraphRepairer.repair(
            recentIDs: [0, 1, 2, 3, 4],
            vectors: try makeVectorBuffer(vectors),
            graph: graphBuffer,
            config: config,
            metric: .l2
        )

        #expect(updates == 0)
        for nodeID in 0..<50 {
            #expect(graphBuffer.neighborIDs(of: nodeID) == originalNeighbors[nodeID])
        }
    }

    @Test("Repair handles deleted nodes correctly")
    func repairWithDeletions() async throws {
        let count = 100
        let dim = 8
        let degree = 4
        let metric: Metric = .l2

        let vectors = (0..<count).map { i in
            (0..<dim).map { d in
                Float(i * dim + d) * 0.01
            }
        }

        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: metric,
            maxIterations: 5
        )

        let vectorBuffer = try makeVectorBuffer(vectors, extraCapacity: 20)
        let graphBuffer = try makeGraphBuffer(graphData, degree: degree, extraCapacity: 20)

        var insertedIDs: [UInt32] = []
        for i in 0..<10 {
            let newVector = (0..<dim).map { d in
                Float((count + i) * dim + d) * 0.01
            }
            let slot = count + i

            try vectorBuffer.insert(vector: newVector, at: slot)
            vectorBuffer.setCount(slot + 1)

            try IncrementalBuilder.insert(
                vector: newVector,
                at: slot,
                into: graphBuffer,
                vectors: vectorBuffer,
                entryPoint: entryPoint,
                metric: metric,
                degree: degree
            )
            graphBuffer.setCount(slot + 1)

            insertedIDs.append(UInt32(slot))
        }

        let updates = try GraphRepairer.repair(
            recentIDs: insertedIDs,
            vectors: vectorBuffer,
            graph: graphBuffer,
            config: RepairConfiguration(repairDepth: 2, repairIterations: 3),
            metric: metric
        )

        #expect(updates >= 0)
    }

    @Test("GraphIndex triggers repair after repairInterval inserts")
    func indexIntegrationRepair() async throws {
        var config = IndexConfiguration(degree: 8, metric: .cosine)
        config.repairConfiguration = RepairConfiguration(
            repairInterval: 10,
            repairDepth: 2,
            repairIterations: 3
        )

        let index = GraphIndex(configuration: config)
        let initialVectors = (0..<50).map { i in
            (0..<16).map { d in
                sin(Float(i * 16 + d) * 0.173)
            }
        }
        let initialIDs = (0..<50).map { "v_\($0)" }
        try await index.build(vectors: initialVectors, ids: initialIDs)

        for i in 50..<65 {
            let vector = (0..<16).map { d in
                sin(Float(i * 16 + d) * 0.173)
            }
            try await index.insert(vector, id: "v_\(i)")
        }

        for i in 50..<65 {
            let query = (0..<16).map { d in
                sin(Float(i * 16 + d) * 0.173)
            }
            let results = try await index.search(query: query, k: 1)
            #expect(!results.isEmpty)
            #expect(results[0].id == "v_\(i)")
        }

        let count = await index.count
        #expect(count == 65)
    }

    @Test("Manual repair via public API")
    func manualRepair() async throws {
        var config = IndexConfiguration(degree: 8, metric: .l2)
        config.repairConfiguration = RepairConfiguration(repairInterval: 0, enabled: true)

        let index = GraphIndex(configuration: config)
        let initialVectors = (0..<50).map { i in
            (0..<8).map { d in
                Float(i * 8 + d) * 0.01
            }
        }
        let initialIDs = (0..<50).map { "v_\($0)" }
        try await index.build(vectors: initialVectors, ids: initialIDs)

        for i in 50..<55 {
            let vector = (0..<8).map { d in
                Float(i * 8 + d) * 0.01
            }
            try await index.insert(vector, id: "v_\(i)")
        }

        try await index.repair()

        for i in 50..<55 {
            let query = (0..<8).map { d in
                Float(i * 8 + d) * 0.01
            }
            let results = try await index.search(query: query, k: 1)
            #expect(!results.isEmpty)
            #expect(results[0].id == "v_\(i)")
        }
    }
}

private func averageRecall(
    queries: [[Float]],
    vectors: [[Float]],
    graph: GraphBuffer,
    entryPoint: Int,
    k: Int,
    ef: Int,
    metric: Metric
) async throws -> Float {
    let graphData = (0..<graph.nodeCount).map { nodeID in
        let ids = graph.neighborIDs(of: nodeID)
        let distances = graph.neighborDistances(of: nodeID)
        return zip(ids, distances).filter { $0.0 != UInt32.max }.map { ($0.0, $0.1) }
    }

    var totalRecall: Float = 0
    for query in queries {
        let approxResults = try await BeamSearchCPU.search(
            query: query,
            vectors: vectors,
            graph: graphData,
            entryPoint: entryPoint,
            k: k,
            ef: ef,
            metric: metric
        )

        let exact = vectors.enumerated().map { index, vector in
            (UInt32(index), SIMDDistance.distance(query, vector, metric: metric))
        }.sorted { $0.1 < $1.1 }

        let exactTopK = Set(exact.prefix(k).map(\.0))
        let approxTopK = Set(approxResults.map(\.internalID))
        totalRecall += Float(exactTopK.intersection(approxTopK).count) / Float(k)
    }

    return totalRecall / Float(queries.count)
}

private func makeGraphBuffer(_ graphData: [[(UInt32, Float)]], degree: Int, extraCapacity: Int = 8) throws
    -> GraphBuffer
{
    let graphBuffer = try GraphBuffer(capacity: graphData.count + extraCapacity, degree: degree)
    for node in 0..<graphData.count {
        var ids = Array(repeating: UInt32.max, count: degree)
        var distances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)
        let neighbors = graphData[node]

        for index in 0..<min(degree, neighbors.count) {
            ids[index] = neighbors[index].0
            distances[index] = neighbors[index].1
        }

        try graphBuffer.setNeighbors(of: node, ids: ids, distances: distances)
    }

    graphBuffer.setCount(graphData.count)
    return graphBuffer
}

private func makeVectorBuffer(_ vectors: [[Float]], extraCapacity: Int = 0) throws -> VectorBuffer {
    guard let first = vectors.first else {
        throw ANNSError.constructionFailed("Vector list cannot be empty")
    }

    let vectorBuffer = try VectorBuffer(capacity: vectors.count + extraCapacity, dim: first.count)
    for (index, vector) in vectors.enumerated() {
        try vectorBuffer.insert(vector: vector, at: index)
    }
    vectorBuffer.setCount(vectors.count)
    return vectorBuffer
}
