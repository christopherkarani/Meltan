import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("IVFPQ Persistence Tests")
struct IVFPQPersistenceTests {
    @Test("Save then load preserves count and search behavior")
    func saveThenLoad() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 6
        )
        let index = try Advanced.IVFPQIndex(capacity: 2_000, dimension: 128, config: config)
        let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 301)
        let training = sampleClusteredVectors(count: 1_000, centers: centers, seed: 302)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 500, centers: centers, seed: 303)
        let ids = (0..<database.count).map { "persist-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let query = sampleClusteredVectors(count: 1, centers: centers, seed: 304)[0]
        let before = await index.search(query: query, k: 20, nprobe: 8)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ivfpq-save-load.bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try await index.save(to: url.path)
        let loaded = try await Advanced.IVFPQIndex.load(from: url.path)

        let loadedCount = await loaded.count
        let originalCount = await index.count
        #expect(loadedCount == originalCount)

        let after = await loaded.search(query: query, k: 20, nprobe: 8)
        #expect(after.count == before.count)
        #expect(after.map(\.id) == before.map(\.id))
    }

    @Test("Round-trip save/load preserves ranked results")
    func roundTripAccuracy() async throws {
        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )
        let index = try Advanced.IVFPQIndex(capacity: 2_500, dimension: 128, config: config)
        let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 401)
        let training = sampleClusteredVectors(count: 1_200, centers: centers, seed: 402)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 700, centers: centers, seed: 403)
        let ids = (0..<database.count).map { "rt-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let queries = sampleClusteredVectors(count: 8, centers: centers, seed: 404)
        var before: [[SearchResult]] = []
        before.reserveCapacity(queries.count)
        for query in queries {
            before.append(await index.search(query: query, k: 15, nprobe: 8))
        }

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ivfpq-round-trip.bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try await index.save(to: url.path)
        let loaded = try await Advanced.IVFPQIndex.load(from: url.path)
        var after: [[SearchResult]] = []
        after.reserveCapacity(queries.count)
        for query in queries {
            after.append(await loaded.search(query: query, k: 15, nprobe: 8))
        }

        #expect(after.count == before.count)
        for idx in 0..<before.count {
            #expect(after[idx].map { $0.id } == before[idx].map { $0.id })
            #expect(after[idx].count == before[idx].count)
            for scoreIndex in 0..<before[idx].count {
                #expect(abs(after[idx][scoreIndex].score - before[idx][scoreIndex].score) < 1e-6)
            }
        }
    }
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
