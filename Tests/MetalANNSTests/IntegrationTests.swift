import Foundation
import Metal
import Testing
@testable import MetalANNS

@Suite("Phase 6 Integration Tests")
struct IntegrationTests {
    @Test("Full lifecycle integration")
    func fullLifecycleIntegration() async throws {
        let baseCount = 500
        let dim = 64
        let k = 10

        let vectors = makeVectors(count: baseCount, dim: dim, seedOffset: 0)
        let ids = (0..<baseCount).map { "v_\($0)" }

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .cosine, maxIterations: 15)
        )
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<20 {
            let query = vectors[(i * 17) % baseCount]
            let results = try await index.search(query: query, k: k)
            #expect(results.count == k)
            #expect(results.allSatisfy { !$0.id.isEmpty })
        }

        let inserted = makeVectors(count: 50, dim: dim, seedOffset: 100_000)
        for i in 0..<50 {
            try await index.insert(inserted[i], id: "new_\(i)")
        }

        for i in 0..<5 {
            let results = try await index.search(query: inserted[i], k: 5)
            #expect(results.contains(where: { $0.id == "new_\(i)" }))
        }

        for i in 0..<10 {
            try await index.delete(id: "v_\(i)")
        }

        let postDelete = try await index.search(query: vectors[42], k: 100)
        let postDeleteIDs = Set(postDelete.map(\.id))
        for i in 0..<10 {
            #expect(!postDeleteIDs.contains("v_\(i)"))
        }

        let saveURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-integration-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let metaURL = URL(fileURLWithPath: saveURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: saveURL)
            try? FileManager.default.removeItem(at: metaURL)
        }

        let baselineLoadedQuery = try await index.search(query: vectors[123], k: 10)
        try await index.save(to: saveURL)
        let loaded = try await GraphIndex.load(from: saveURL)
        let loadedQuery = try await loaded.search(query: vectors[123], k: 10)

        #expect(baselineLoadedQuery.map(\.id) == loadedQuery.map(\.id))

        let expectedCount = baseCount + 50 - 10
        let activeCount = await index.count
        let loadedCount = await loaded.count
        #expect(activeCount == expectedCount)
        #expect(loadedCount == expectedCount)
    }

    @Test("Recall at 10 over threshold")
    func recallAtTenOverNinetyPercent() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let vectorCount = 500
        let dim = 32
        let k = 10

        let vectors = makeVectors(count: vectorCount, dim: dim, seedOffset: 0)
        let ids = (0..<vectorCount).map { "v_\($0)" }

        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 16,
                metric: .cosine,
                efSearch: 64,
                maxIterations: 15
            )
        )
        try await index.build(vectors: vectors, ids: ids)

        let queries = makeVectors(count: 50, dim: dim, seedOffset: 500_000)
        var totalRecall: Float = 0

        for query in queries {
            let approx = try await index.search(query: query, k: k)
            let approxIDs = Set(approx.map(\.id))
            let exact = bruteForceTopK(query: query, vectors: vectors, ids: ids, k: k)
            let exactIDs = Set(exact)
            totalRecall += Float(approxIDs.intersection(exactIDs).count) / Float(k)
        }

        let recallAtTen = totalRecall / Float(queries.count)
        #expect(recallAtTen > 0.90)
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    private func bruteForceTopK(query: [Float], vectors: [[Float]], ids: [String], k: Int) -> [String] {
        let scored = vectors.enumerated().map { index, vector -> (id: String, distance: Float) in
            (ids[index], cosineDistance(lhs: query, rhs: vector))
        }
        return scored.sorted { $0.distance < $1.distance }.prefix(k).map(\.id)
    }

    private func cosineDistance(lhs: [Float], rhs: [Float]) -> Float {
        var dot: Float = 0
        var lhsNorm: Float = 0
        var rhsNorm: Float = 0

        for i in 0..<lhs.count {
            dot += lhs[i] * rhs[i]
            lhsNorm += lhs[i] * lhs[i]
            rhsNorm += rhs[i] * rhs[i]
        }

        let denom = sqrt(lhsNorm) * sqrt(rhsNorm)
        return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
    }
}
