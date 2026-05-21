import Foundation
import Testing
@testable import MetalANNSCore

@Suite("Incremental Insert Tests")
struct IncrementalTests {
    @Test("Insert and find new vectors")
    func insertAndFindNew() async throws {
        let initialCount = 100
        let dim = 8
        let degree = 4
        let metric: Metric = .cosine

        var vectors = seededVectors(count: initialCount, dim: dim)
        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: metric,
            maxIterations: 10
        )

        let vectorBuffer = try vectorBufferFrom(vectors)
        let graphBuffer = try graphBufferFrom(graphData, degree: degree)

        for newIndex in 0..<10 {
            let query = seededVector(dim: dim, seed: 1000 + UInt64(newIndex))
            let internalID = initialCount + newIndex

            try vectorBuffer.insert(vector: query, at: internalID)
            try IncrementalBuilder.insert(
                vector: query,
                at: internalID,
                into: graphBuffer,
                vectors: vectorBuffer,
                entryPoint: entryPoint,
                metric: metric,
                degree: degree
            )

            vectors.append(query)
            graphBuffer.setCount(internalID + 1)
            vectorBuffer.setCount(internalID + 1)

            let graphForSearch = graphFrom(graphBuffer)
            let searchResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graphForSearch,
                entryPoint: Int(entryPoint),
                k: 1,
                ef: degree * 8,
                metric: metric
            )

            #expect(!searchResults.isEmpty)
            #expect(searchResults[0].internalID == UInt32(internalID))
            #expect(abs(searchResults[0].score) < 1e-4)
        }
    }

    @Test("Recall degradation stays below 5%")
    func insertRecallDegradation() async throws {
        let initialCount = 200
        let dim = 16
        let degree = 8
        let metric: Metric = .cosine

        var vectors = seededVectors(count: initialCount, dim: dim)
        let (initialGraphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: metric,
            maxIterations: 10
        )

        let vectorBuffer = try vectorBufferFrom(vectors)
        let graphBuffer = try graphBufferFrom(initialGraphData, degree: degree)

        let queryVectors = Array(vectors.prefix(10))
        let baselineRecall = try await averageRecall(
            queries: queryVectors,
            vectors: vectors,
            graph: graphFrom(graphBuffer),
            entryPoint: Int(entryPoint),
            k: 5,
            ef: 32,
            metric: metric
        )

        for newIndex in 0..<20 {
            let newVector = seededVector(dim: dim, seed: 2000 + UInt64(newIndex))
            let internalID = initialCount + newIndex

            try vectorBuffer.insert(vector: newVector, at: internalID)
            vectors.append(newVector)

            try IncrementalBuilder.insert(
                vector: newVector,
                at: internalID,
                into: graphBuffer,
                vectors: vectorBuffer,
                entryPoint: entryPoint,
                metric: metric,
                degree: degree
            )

            graphBuffer.setCount(internalID + 1)
            vectorBuffer.setCount(internalID + 1)
        }

        let postRecall = try await averageRecall(
            queries: queryVectors,
            vectors: vectors,
            graph: graphFrom(graphBuffer),
            entryPoint: Int(entryPoint),
            k: 5,
            ef: 32,
            metric: metric
        )

        #expect(postRecall > baselineRecall - 0.05)
    }

    private func averageRecall(
        queries: [[Float]],
        vectors: [[Float]],
        graph: [[(UInt32, Float)]],
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> Float {
        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        var totalRecall: Float = 0

        for query in queries {
            let approxResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: metric
            )

            let exact = try await withVectorBuffer(flat) { pointer in
                try await backend.computeDistances(
                    query: query,
                    vectors: pointer,
                    vectorCount: vectors.count,
                    dim: vectors[0].count,
                    metric: metric
                )
            }

            let exactTopK = Set(
                exact.enumerated()
                    .sorted { $0.element < $1.element }
                    .prefix(k)
                    .map { UInt32($0.offset) }
            )
            let approxTopK = Set(approxResults.map(\.internalID))
            totalRecall += Float(exactTopK.intersection(approxTopK).count) / Float(k)
        }

        return totalRecall / Float(queries.count)
    }

    private func vectorBufferFrom(_ vectors: [[Float]]) throws -> VectorBuffer {
        guard let first = vectors.first else {
            throw ANNSError.constructionFailed("Vector list cannot be empty")
        }

        let vectorBuffer = try VectorBuffer(capacity: vectors.count + 20, dim: first.count)
        for (index, vector) in vectors.enumerated() {
            try vectorBuffer.insert(vector: vector, at: index)
        }
        vectorBuffer.setCount(vectors.count)
        return vectorBuffer
    }

    private func graphBufferFrom(_ graphData: [[(UInt32, Float)]], degree: Int) throws -> GraphBuffer {
        let graphBuffer = try GraphBuffer(capacity: graphData.count + 20, degree: degree)
        for node in 0..<graphData.count {
            let neighbors = graphData[node]
            var ids = Array(repeating: UInt32.max, count: degree)
            var distances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)

            for index in 0..<min(degree, neighbors.count) {
                ids[index] = neighbors[index].0
                distances[index] = neighbors[index].1
            }

            try graphBuffer.setNeighbors(of: node, ids: ids, distances: distances)
        }
        graphBuffer.setCount(graphData.count)
        return graphBuffer
    }

    private func graphFrom(_ graph: GraphBuffer) -> [[(UInt32, Float)]] {
        (0..<graph.nodeCount).map { index in
            let ids = graph.neighborIDs(of: index)
            let distances = graph.neighborDistances(of: index)
            return Array(zip(ids, distances))
        }
    }
}

private func withVectorBuffer<T>(
    _ values: [Float],
    _ body: (UnsafeBufferPointer<Float>) async throws -> T
) async throws -> T {
    let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
    buffer.initialize(from: values)
    defer {
        buffer.deinitialize()
        buffer.deallocate()
    }
    return try await body(UnsafeBufferPointer(buffer))
}
