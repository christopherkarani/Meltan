import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Disk-Backed Index Tests")
struct DiskBackedTests {
    @Test("Disk-backed search produces correct results")
    func diskBackedSearchWorks() async throws {
        let dim = 32
        let vectors = makeVectors(count: 200, dim: dim, seedOffset: 400)
        let ids = (0..<200).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-disk-backed-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let tempMetaURL = URL(fileURLWithPath: tempURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
        }

        try await index.save(to: tempURL)

        let diskBacked = try await GraphIndex.loadDiskBacked(from: tempURL)
        let normal = try await GraphIndex.load(from: tempURL)

        // Disk-backed search routes through graph traversal (its staged
        // windows are not flat-scan eligible), so results are approximate by
        // design. Assert high recall against brute force rather than strict
        // equality between two independently built graph traversals.
        var recallSum = 0.0
        var scoreChecks = 0
        for query in vectors.prefix(10) {
            let truth = Set(
                bruteForceTopK(query: query, vectors: vectors, k: 10)
                    .map { "v\($0)" })
            let diskResults = try await diskBacked.search(query: query, k: 10)
            let normalResults = try await normal.search(query: query, k: 10)

            #expect(diskResults.count == 10)
            #expect(normalResults.count == 10)

            // In-memory load serves the fused exact scan; it must be perfect.
            let normalHits = Set(normalResults.map(\.id)).intersection(truth).count
            #expect(normalHits == 10, "in-memory load recall below exact: \(normalHits)/10")

            recallSum += Double(Set(diskResults.map(\.id)).intersection(truth).count) / 10.0

            for pair in zip(diskResults, normalResults) {
                #expect(pair.0.score >= -1e-5 && pair.0.score <= 2.0 + 1e-5)
                #expect(pair.1.score >= -1e-5 && pair.1.score <= 2.0 + 1e-5)
                scoreChecks += 1
            }
        }
        let meanRecall = recallSum / 10.0
        #expect(meanRecall >= 0.85, "disk-backed mean recall@10 too low: \(meanRecall)")
        #expect(scoreChecks == 100)
    }

    private func bruteForceTopK(query: [Float], vectors: [[Float]], k: Int) -> [Int] {
        let scored = vectors.enumerated().map { index, vector -> (Int, Float) in
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            let distance = denom < 1e-10 ? Float(1.0) : (1.0 - dot / denom)
            return (index, distance)
        }
        return Array(scored.sorted { $0.1 < $1.1 }.prefix(k).map(\.0))
    }

    @Test("Disk-backed load works with v3 mmap format")
    func diskBackedWorksWithV3() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 401)
        let ids = (0..<100).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-disk-backed-v3-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let tempMetaURL = URL(fileURLWithPath: tempURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
        }

        try await index.saveMmapCompatible(to: tempURL)
        let diskBacked = try await GraphIndex.loadDiskBacked(from: tempURL)

        let results = try await diskBacked.search(query: vectors[3], k: 10)
        #expect(!results.isEmpty)
        #expect(results[0].id == "v3")
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.057) + cos(i * 0.033)
            }
        }
    }
}
