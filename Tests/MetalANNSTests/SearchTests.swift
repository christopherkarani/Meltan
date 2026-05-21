import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("Search Tests")
struct SearchTests {
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

    @Test("CPU beam search returns k results")
    func cpuSearchReturnsK() async throws {
        let n = 100
        let dim = 16
        let degree = 8
        let k = 5
        let vectors = seededVectors(count: n, dim: dim)

        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: 10
        )

        let query = seededVector(dim: dim)
        let results = try await BeamSearchCPU.search(
            query: query,
            vectors: vectors,
            graph: graphData,
            entryPoint: Int(entryPoint),
            k: k,
            ef: 32,
            metric: .cosine
        )

        #expect(results.count == k)
        for index in 1..<results.count {
            #expect(results[index].score >= results[index - 1].score)
        }
    }

    @Test("CPU search recall > 0.90 on 1000 vectors")
    func cpuSearchRecall() async throws {
        let n = 1000
        let dim = 32
        let degree = 16
        let k = 10
        let ef = 64
        let queryCount = 20
        let vectors = seededVectors(count: n, dim: dim)

        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: 15
        )

        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        var totalRecall: Float = 0

        for _ in 0..<queryCount {
            let query = seededVector(dim: dim)
            let results = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graphData,
                entryPoint: Int(entryPoint),
                k: k,
                ef: ef,
                metric: .cosine
            )

            let exactDistances = try await withVectorBuffer(flat) { pointer in
                try await backend.computeDistances(
                    query: query,
                    vectors: pointer,
                    vectorCount: n,
                    dim: dim,
                    metric: .cosine
                )
            }

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
        #expect(averageRecall > 0.90, "Average recall \\(averageRecall) below 0.90")
    }
}
