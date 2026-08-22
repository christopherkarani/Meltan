import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Binary Quantization Tests")
struct BinaryQuantizationTests {
    @Test("binary storage provides expected memory reduction")
    func memoryReduction() throws {
        let binaryBuffer = try BinaryVectorBuffer(capacity: 1_000, dim: 128)
        let floatBuffer = try VectorBuffer(capacity: 1_000, dim: 128)

        #expect(binaryBuffer.buffer.length * 32 <= floatBuffer.buffer.length)
    }

    @Test("pack and unpack round-trip remains stable")
    func packUnpackRoundTrip() throws {
        let dim = 32
        let vectorCount = 10
        let buffer = try BinaryVectorBuffer(capacity: vectorCount, dim: dim)

        for index in 0..<vectorCount {
            let vector = seededVector(dim: dim)
            try buffer.insert(vector: vector, at: index)

            let unpacked = buffer.vector(at: index)
            #expect(unpacked.allSatisfy { $0 == 0.0 || $0 == 1.0 })

            let repacked = packLikeBuffer(unpacked)
            #expect(repacked == buffer.packedVector(at: index))
        }
    }

    @Test("dimension must be divisible by 8")
    func dimensionConstraintEnforced() throws {
        do {
            _ = try BinaryVectorBuffer(capacity: 10, dim: 7)
            #expect(Bool(false), "Expected construction failure for dim not divisible by 8")
        } catch let error as ANNSError {
            if case .constructionFailed = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.constructionFailed but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.constructionFailed")
        }
    }

    @Test("hamming distance correctness")
    func hammingDistanceCorrectness() {
        let a: [Float] = [1, 0, 1, 0, 1, 1, 0, 0]
        let b: [Float] = [0, 0, 1, 1, 1, 0, 0, 1]

        #expect(SIMDDistance.hamming(a, b) == 4.0)
        #expect(SIMDDistance.distance(a, b, metric: .hamming) == 4.0)
    }

    @Test("binary insert and search")
    func binaryInsertAndSearch() async throws {
        let dim = 64
        let vectors = randomBinaryVectors(count: 200, dim: dim)
        let ids = (0..<vectors.count).map { "v_\($0)" }

        let config = IndexConfiguration(
            degree: 8,
            metric: .hamming,
            efSearch: 96,
            useBinary: true
        )
        let index = GraphIndex(configuration: config)
        try await index.build(vectors: vectors, ids: ids)

        for query in vectors.prefix(5) {
            let results = try await index.search(query: query, k: 10)
            #expect(results.count == 10)
            #expect(results.allSatisfy { $0.internalID < UInt32(vectors.count) })
        }
    }

    @Test("hamming recall")
    func hammingRecall() async throws {
        let dim = 64
        let vectors = randomBinaryVectors(count: 500, dim: dim)
        let ids = (0..<vectors.count).map { "v_\($0)" }

        let config = IndexConfiguration(
            degree: 16,
            metric: .hamming,
            efSearch: 128,
            useBinary: true
        )
        let index = GraphIndex(configuration: config)
        try await index.build(vectors: vectors, ids: ids)

        let quantizedVectors = vectors.map(quantizeToBits)
        let queries = randomBinaryVectors(count: 20, dim: dim)

        var recallSum = 0.0
        for query in queries {
            let quantizedQuery = quantizeToBits(query)
            let groundTruth = bruteForceHammingTopK(query: quantizedQuery, vectors: quantizedVectors, k: 10)
            let results = try await index.search(query: query, k: 10)
            recallSum += recall(results: results, groundTruth: groundTruth)
        }

        let meanRecall = recallSum / Double(queries.count)
        #expect(meanRecall >= 0.70)
    }

    @Test("persistence round-trip")
    func persistenceRoundTrip() async throws {
        let dim = 64
        let vectors = randomBinaryVectors(count: 100, dim: dim)
        let ids = (0..<vectors.count).map { "v_\($0)" }

        let config = IndexConfiguration(
            degree: 8,
            metric: .hamming,
            efSearch: 96,
            useBinary: true
        )
        let index = GraphIndex(configuration: config)
        try await index.build(vectors: vectors, ids: ids)

        let query = vectors[7]
        let before = try await index.search(query: query, k: 5)

        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-binary-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let metaURL = URL(fileURLWithPath: fileURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: metaURL)
        }

        try await index.save(to: fileURL)
        let loaded = try await GraphIndex.load(from: fileURL)
        let after = try await loaded.search(query: query, k: 5)

        let beforeIDs = before.map(\.id)
        let afterIDs = after.map(\.id)
        #expect(beforeIDs.count == afterIDs.count)
        #expect(beforeIDs.first == afterIDs.first)
        let overlap = Set(beforeIDs).intersection(afterIDs).count
        #expect(overlap >= min(4, beforeIDs.count))
    }

    @Test("useBinary build enforces dimension divisibility")
    func buildDimensionConstraintEnforced() async throws {
        let vectors = randomBinaryVectors(count: 16, dim: 8).map { Array($0.prefix(7)) }
        let ids = (0..<vectors.count).map { "bad_\($0)" }
        let config = IndexConfiguration(metric: .hamming, useBinary: true)
        let index = GraphIndex(configuration: config)

        do {
            try await index.build(vectors: vectors, ids: ids)
            #expect(Bool(false), "Expected construction failure for dim not divisible by 8")
        } catch let error as ANNSError {
            if case .constructionFailed = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.constructionFailed but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.constructionFailed")
        }
    }

    @Test("useBinary requires hamming metric")
    func useBinaryRequiresHammingMetric() async throws {
        let dim = 64
        let vectors = randomBinaryVectors(count: 32, dim: dim)
        let ids = (0..<vectors.count).map { "v_\($0)" }

        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 8,
                metric: .cosine,
                efSearch: 96,
                useBinary: true
            )
        )

        do {
            try await index.build(vectors: vectors, ids: ids)
            #expect(Bool(false), "Expected construction failure when useBinary is paired with non-hamming metric")
        } catch let error as ANNSError {
            if case .constructionFailed = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.constructionFailed but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.constructionFailed")
        }
    }

    @Test("hamming metric requires binary storage")
    func hammingMetricRequiresBinaryStorage() async throws {
        let vectors = randomBinaryVectors(count: 32, dim: 64)
        let ids = (0..<vectors.count).map { "v_\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(metric: .hamming, useBinary: false))
        do {
            try await index.build(vectors: vectors, ids: ids)
            #expect(Bool(false), "Expected construction failure for hamming without useBinary")
        } catch let error as ANNSError {
            if case .constructionFailed = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.constructionFailed but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.constructionFailed")
        }
    }

    @Test("compaction preserves binary storage semantics")
    func compactionPreservesBinaryStorageSemantics() async throws {
        let vectors = randomBinaryVectors(count: 120, dim: 64)
        let ids = (0..<vectors.count).map { "v_\($0)" }
        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 8, metric: .hamming, efSearch: 96, useBinary: true)
        )
        try await index.build(vectors: vectors, ids: ids)
        try await index.delete(id: ids[0])
        try await index.delete(id: ids[1])
        try await index.compact()

        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-binary-compact-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let metaURL = URL(fileURLWithPath: fileURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: metaURL)
        }

        try await index.save(to: fileURL)
        let loaded = try await GraphIndex.load(from: fileURL)

        let results = try await loaded.search(query: vectors[10], k: 10)
        #expect(!results.isEmpty)
    }

    @Test("mmap and disk-backed loaders support binary storage")
    func mmapAndDiskBackedSupportBinaryStorage() async throws {
        let vectors = randomBinaryVectors(count: 100, dim: 64)
        let ids = (0..<vectors.count).map { "v_\($0)" }
        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 8, metric: .hamming, efSearch: 96, useBinary: true)
        )
        try await index.build(vectors: vectors, ids: ids)

        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-binary-mmap-disk-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let metaURL = URL(fileURLWithPath: fileURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: metaURL)
        }

        try await index.saveMmapCompatible(to: fileURL)
        let mmapLoaded = try await GraphIndex.loadMmap(from: fileURL)
        let diskLoaded = try await GraphIndex.loadDiskBacked(from: fileURL)
        let query = vectors[8]

        let mmapResults = try await mmapLoaded.search(query: query, k: 5)
        let diskResults = try await diskLoaded.search(query: query, k: 5)
        #expect(!mmapResults.isEmpty)
        #expect(!diskResults.isEmpty)
    }

    private func packLikeBuffer(_ vector: [Float]) -> [UInt8] {
        let bytesPerVector = vector.count / 8
        var packed = [UInt8](repeating: 0, count: bytesPerVector)

        for byteIndex in 0..<bytesPerVector {
            var byte: UInt8 = 0
            for bit in 0..<8 {
                let dimIndex = byteIndex * 8 + bit
                if vector[dimIndex] > 0.5 {
                    byte |= (1 << (7 - bit))
                }
            }
            packed[byteIndex] = byte
        }

        return packed
    }

    private func randomBinaryVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dim).map { _ in Bool.random() ? 1.0 : -1.0 }
        }
    }

    private func quantizeToBits(_ vector: [Float]) -> [Float] {
        vector.map { $0 >= 0 ? 1.0 : 0.0 }
    }

    private func bruteForceHammingTopK(query: [Float], vectors: [[Float]], k: Int) -> Set<Int> {
        Set(
            vectors.enumerated()
                .map { index, vector in (index, SIMDDistance.hamming(query, vector)) }
                .sorted { lhs, rhs in
                    if lhs.1 == rhs.1 {
                        return lhs.0 < rhs.0
                    }
                    return lhs.1 < rhs.1
                }
                .prefix(k)
                .map(\.0)
        )
    }

    private func recall(results: [SearchResult], groundTruth: Set<Int>) -> Double {
        let predicted = Set(results.map { Int($0.internalID) })
        let overlap = predicted.intersection(groundTruth).count
        return Double(overlap) / Double(max(1, groundTruth.count))
    }
}
