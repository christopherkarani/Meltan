import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Memory-Mapped I/O Tests")
struct MmapTests {
    @Test("Mmap load produces same search results as normal load")
    func mmapProducesSameResults() async throws {
        let dim = 32
        let vectors = makeVectors(count: 200, dim: dim, seedOffset: 0)
        let ids = (0..<200).map { "vec_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-mmap-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try await index.saveMmapCompatible(to: tempURL)

        let mmapLoaded = try await GraphIndex.loadMmap(from: tempURL)
        let normalLoaded = try await GraphIndex.load(from: tempURL)

        for query in vectors.prefix(10) {
            let mmapResults = try await mmapLoaded.search(query: query, k: 10)
            let normalResults = try await normalLoaded.search(query: query, k: 10)

            #expect(mmapResults.map(\.id) == normalResults.map(\.id))
            #expect(mmapResults.map(\.internalID) == normalResults.map(\.internalID))
        }

        let extraVector = makeVectors(count: 1, dim: dim, seedOffset: 999_999)[0]
        try await normalLoaded.insert(extraVector, id: "extra_after_normal_load")
        let inserted = try await normalLoaded.search(query: extraVector, k: 5)
        #expect(inserted.contains { $0.id == "extra_after_normal_load" })
    }

    @Test("Mmap save and load roundtrip preserves all data")
    func mmapRoundtrip() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 500)
        let ids = (0..<100).map { "vec_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)
        try await index.delete(id: ids[0])
        try await index.delete(id: ids[1])

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-mmap-roundtrip-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try await index.saveMmapCompatible(to: tempURL)
        let loaded = try await GraphIndex.loadMmap(from: tempURL)
        let loadedNormal = try await GraphIndex.load(from: tempURL)

        #expect(await loaded.count == 98)
        #expect(await loadedNormal.count == 98)

        for i in 2..<10 {
            let results = try await loaded.search(query: vectors[i], k: 5)
            #expect(results.contains { $0.id == ids[i] })
        }

        let mmapFiltered = try await loaded.search(query: vectors[0], k: 10)
        #expect(!mmapFiltered.contains { $0.id == ids[0] })
        #expect(!mmapFiltered.contains { $0.id == ids[1] })

        let filtered = try await loadedNormal.search(query: vectors[0], k: 10)
        #expect(!filtered.contains { $0.id == ids[0] })
        #expect(!filtered.contains { $0.id == ids[1] })

        do {
            try await loaded.insert(vectors[0], id: "should_fail")
            #expect(Bool(false), "Expected read-only insert failure for mmap-loaded index")
        } catch let error as ANNSError {
            if case .constructionFailed = error { }
            else {
                #expect(Bool(false), "Expected ANNSError.constructionFailed but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.constructionFailed")
        }
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.193) + cos(i * 0.057)
            }
        }
    }
}
