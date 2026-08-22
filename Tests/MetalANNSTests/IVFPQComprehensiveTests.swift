import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("IVFPQ Comprehensive Tests")
struct IVFPQComprehensiveTests {
    @Test("benchmarkSearchThroughput")
    func benchmarkSearchThroughput() async throws {
        let scenario = try await makeScenario(seed: 501, trainingCount: 1_000, databaseCount: 1_200)
        let queries = sampleClusteredVectors(count: 60, centers: scenario.centers, seed: 504)

        let start = ContinuousClock.now
        for query in queries {
            _ = await scenario.index.search(query: query, k: 10, nprobe: 8)
        }
        let elapsedDuration = start.duration(to: .now).components
        let elapsedSeconds = Double(elapsedDuration.seconds) + (Double(elapsedDuration.attoseconds) / 1e18)
        let qps = Float(Double(queries.count) / max(elapsedSeconds, 1e-6))

        print("IVFPQ benchmarkSearchThroughput: \(qps) queries/sec")
        #expect(qps > 10)
    }

    @Test("benchmarkMemoryUsage")
    func benchmarkMemoryUsage() async throws {
        let scenario = try await makeScenario(seed: 601, trainingCount: 1_000, databaseCount: 1_500)
        let compressedVectorBytes = await scenario.index.estimatedVectorCodeBytes()
        let totalIndexBytes = await scenario.index.estimatedMemoryBytes()
        let uncompressedVectorBytes = scenario.database.count * 128 * MemoryLayout<Float>.stride

        let vectorReduction = Float(uncompressedVectorBytes) / Float(compressedVectorBytes)
        let totalReduction = Float(uncompressedVectorBytes) / Float(totalIndexBytes)
        let totalMB = Float(totalIndexBytes) / (1024 * 1024)

        print(
            "IVFPQ benchmarkMemoryUsage: totalMB=\(totalMB), vectorReduction=\(vectorReduction)x, totalReduction=\(totalReduction)x"
        )
        #expect(vectorReduction > 30)
    }

    @Test("benchmarkRecallVsQPS")
    func benchmarkRecallVsQPS() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )
        let index = try Advanced.IVFPQIndex(capacity: 2_000, dimension: 128, config: config)
        let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 31)
        let training = sampleClusteredVectors(count: 1_200, centers: centers, seed: 32)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 400, centers: centers, seed: 47)
        let ids = (0..<database.count).map { "rvq-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let queries = sampleClusteredVectors(count: 30, centers: centers, seed: 59)

        let recall1 = await averageRecallAt10(
            index: index,
            databaseVectors: database,
            databaseIDs: ids,
            queries: queries,
            nprobe: 1
        )
        let recall4 = await averageRecallAt10(
            index: index,
            databaseVectors: database,
            databaseIDs: ids,
            queries: queries,
            nprobe: 4
        )
        let recall8 = await averageRecallAt10(
            index: index,
            databaseVectors: database,
            databaseIDs: ids,
            queries: queries,
            nprobe: 8
        )
        let recall16 = await averageRecallAt10(
            index: index,
            databaseVectors: database,
            databaseIDs: ids,
            queries: queries,
            nprobe: 16
        )

        print(
            "IVFPQ benchmarkRecallVsQPS: nprobe=1 \(recall1), nprobe=4 \(recall4), nprobe=8 \(recall8), nprobe=16 \(recall16)"
        )

        #expect(recall4 >= recall1 - 0.01)
        #expect(recall8 >= recall4 - 0.01)
        #expect(recall16 >= recall8 - 0.01)
        #expect(recall8 > 0.80)
    }
}

private struct IVFPQScenario {
    let index: Advanced.IVFPQIndex
    let centers: [[Float]]
    let database: [[Float]]
    let ids: [String]
}

private func makeScenario(seed: UInt64, trainingCount: Int, databaseCount: Int) async throws -> IVFPQScenario {
    let config = IVFPQConfiguration(
        numSubspaces: 8,
        numCentroids: 256,
        numCoarseCentroids: 32,
        nprobe: 8,
        metric: .l2,
        trainingIterations: 8
    )
    let index = try Advanced.IVFPQIndex(capacity: trainingCount + databaseCount + 100, dimension: 128, config: config)

    let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: seed)
    let training = sampleClusteredVectors(count: trainingCount, centers: centers, seed: seed + 1)
    try await index.train(vectors: training)

    let database = sampleClusteredVectors(count: databaseCount, centers: centers, seed: seed + 2)
    let ids = (0..<database.count).map { "bench-\($0)" }
    try await index.add(vectors: database, ids: ids)

    return IVFPQScenario(index: index, centers: centers, database: database, ids: ids)
}

private func averageRecallAt10(
    index: Advanced.IVFPQIndex,
    databaseVectors: [[Float]],
    databaseIDs: [String],
    queries: [[Float]],
    nprobe: Int
) async -> Float {
    let k = 10
    var totalRecall: Float = 0

    for query in queries {
        let exactTop = exactTopK(query: query, vectors: databaseVectors, ids: databaseIDs, k: k)
        let approx = await index.search(query: query, k: k, nprobe: nprobe)
        let overlap = Set(exactTop).intersection(Set(approx.map { $0.id })).count
        totalRecall += Float(overlap) / Float(k)
    }

    return totalRecall / Float(queries.count)
}

private func exactTopK(query: [Float], vectors: [[Float]], ids: [String], k: Int) -> [String] {
    let scored = zip(ids, vectors).map { id, vector in
        (id, SIMDDistance.distance(query, vector, metric: .l2))
    }
    return scored.sorted { $0.1 < $1.1 }.prefix(k).map(\.0)
}

private func makeClusterCenters(dimension: Int, clusters: Int, seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<clusters).map { _ in
        (0..<dimension).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
    }
}

private func sampleClusteredVectors(count: Int, centers: [[Float]], seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    let clusters = centers.count
    return (0..<count).map { row in
        let cluster = row % clusters
        return centers[cluster].map { center in
            center + Float.random(in: -0.005...0.005, using: &rng)
        }
    }
}
