import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Advanced.IVFPQIndex Tests")
struct IVFPQIndexTests {
    @Test("Train and add vectors")
    func trainAndAdd() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 6
        )
        let index = try Advanced.IVFPQIndex(capacity: 2_000, dimension: 128, config: config)

        let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 5)
        let training = sampleClusteredVectors(count: 800, centers: centers, seed: 6)
        try await index.train(vectors: training)

        let vectors = sampleClusteredVectors(count: 200, centers: centers, seed: 17)
        let ids = (0..<vectors.count).map { "vec-\($0)" }
        try await index.add(vectors: vectors, ids: ids)

        let count = await index.count
        #expect(count == 200)
    }

    @Test("Recall@10 exceeds 0.80 with default nprobe")
    func searchRecall() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )
        let index = try Advanced.IVFPQIndex(capacity: 3_000, dimension: 128, config: config)

        let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 31)
        let training = sampleClusteredVectors(count: 1_200, centers: centers, seed: 32)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 400, centers: centers, seed: 47)
        let ids = (0..<database.count).map { "db-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let queries = sampleClusteredVectors(count: 30, centers: centers, seed: 59)
        let recall = await averageRecallAt10(
            index: index,
            databaseVectors: database,
            databaseIDs: ids,
            queries: queries
        )

        #expect(recall > 0.80, "Recall@10 was \(recall)")
    }

    @Test("Recall improves as nprobe increases")
    func nprobeEffect() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )
        let index = try Advanced.IVFPQIndex(capacity: 3_000, dimension: 128, config: config)

        let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 71)
        let training = sampleClusteredVectors(count: 1_100, centers: centers, seed: 72)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 500, centers: centers, seed: 73)
        let ids = (0..<database.count).map { "np-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let queries = sampleClusteredVectors(count: 20, centers: centers, seed: 79)
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

        #expect(recall4 >= recall1 - 0.01)
        #expect(recall8 >= recall4 - 0.01)
    }

    @Test("Compressed payload shows 30x+ reduction")
    func memoryFootprint() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 16,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 6
        )
        let index = try Advanced.IVFPQIndex(capacity: 2_000, dimension: 128, config: config)

        let centers = makeClusterCenters(dimension: 128, clusters: 16, seed: 101)
        let training = sampleClusteredVectors(count: 900, centers: centers, seed: 102)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 800, centers: centers, seed: 107)
        let ids = (0..<database.count).map { "mem-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let compressedVectorBytes = await index.estimatedVectorCodeBytes()
        let uncompressedVectorBytes = database.count * 128 * MemoryLayout<Float>.stride
        let reduction = Float(uncompressedVectorBytes) / Float(compressedVectorBytes)

        #expect(reduction > 30, "Observed reduction \(reduction)x")
    }
}

private func averageRecallAt10(
    index: Advanced.IVFPQIndex,
    databaseVectors: [[Float]],
    databaseIDs: [String],
    queries: [[Float]],
    nprobe: Int? = nil
) async -> Float {
    let k = 10
    var total: Float = 0

    for query in queries {
        let exactTop = exactTopK(query: query, vectors: databaseVectors, ids: databaseIDs, k: k)
        let approx = await index.search(query: query, k: k, nprobe: nprobe)
        let overlap = Set(exactTop).intersection(Set(approx.map(\.id))).count
        total += Float(overlap) / Float(k)
    }

    return total / Float(queries.count)
}

private func exactTopK(query: [Float], vectors: [[Float]], ids: [String], k: Int) -> [String] {
    let scored = zip(ids, vectors).map { id, vector in
        (id, SIMDDistance.distance(query, vector, metric: .l2))
    }
    return
        scored
        .sorted { $0.1 < $1.1 }
        .prefix(k)
        .map(\.0)
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
            center + Float.random(in: -0.02...0.02, using: &rng)
        }
    }
}
