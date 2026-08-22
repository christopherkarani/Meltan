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

        for query in vectors.prefix(10) {
            let diskResults = try await diskBacked.search(query: query, k: 10)
            let normalResults = try await normal.search(query: query, k: 10)

            #expect(diskResults.map(\.id) == normalResults.map(\.id))
            #expect(diskResults.count == normalResults.count)
            for (lhs, rhs) in zip(diskResults.map(\.score), normalResults.map(\.score)) {
                #expect(abs(lhs - rhs) < 1e-3)
            }
        }
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
