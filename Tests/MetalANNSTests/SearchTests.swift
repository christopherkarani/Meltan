import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Search Tests")
struct SearchTests {
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
        #expect(averageRecall > 0.90, "Average recall \\(averageRecall) below 0.90")
    }
}
