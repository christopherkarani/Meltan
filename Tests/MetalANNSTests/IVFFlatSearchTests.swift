import Foundation
import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

/// Recall vs brute force for the shipped IVF-flat search (bucket B).
@Suite("IVF-flat approximate search")
struct IVFFlatSearchTests {
    private func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> [UInt32] {
        var scored: [(UInt32, Float)] = []
        scored.reserveCapacity(vectors.count)
        for index in 0..<vectors.count {
            let dist = distance(query: query, vector: vectors[index], metric: metric)
            scored.append((UInt32(index), dist))
        }
        scored.sort { lhs, rhs in
            lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
        }
        return scored.prefix(k).map { $0.0 }
    }

    private func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : 1.0 - (dot / denom)
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        case .hamming:
            return 0
        }
    }

    private func recall(predicted: [UInt32], truth: [UInt32], k: Int) -> Double {
        let pred = Set(predicted.prefix(k))
        let exact = Set(truth.prefix(k))
        guard !exact.isEmpty else { return 1 }
        return Double(pred.intersection(exact).count) / Double(exact.count)
    }

    /// Same generator the 10× claim uses (`BenchmarkRunner.makeVectors`).
    private func benchLikeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    @Test("IVF-flat recall@10 ≥ 0.95 on dim-384 clustered corpus at default nprobe")
    func clusteredDim384RecallAtOperatingPoint() throws {
        let dim = 384
        let count = 1_024
        let k = 10
        let centerCount = 32
        let centers = seededVectors(count: centerCount, dim: dim, seed: 3)
        let rows = seededClusteredVectors(count: count, dim: dim, centers: centers, seed: 11)
        let queries = seededClusteredVectors(
            count: 24,
            dim: dim,
            centers: centers,
            seed: 99
        )

        let buffer = try VectorBuffer(capacity: count, dim: dim)
        try buffer.batchInsert(vectors: rows, startingAt: 0)
        buffer.setCount(count)
        let nlist = IVFFlatSearch.listCount(for: count)
        IVFFlatSearch.prepare(vectors: buffer, metric: .cosine, nlist: nlist)

        var recall10 = 0.0
        for query in queries {
            let results = try IVFFlatSearch.search(
                query: query,
                vectors: buffer,
                k: k,
                nlist: nlist,
                nprobe: IVFFlatSearch.defaultNProbe,
                metric: .cosine
            )
            let predicted = results.map(\.internalID)
            let truth = bruteForceTopK(query: query, vectors: rows, k: k, metric: .cosine)
            recall10 += recall(predicted: predicted, truth: truth, k: k)
        }
        recall10 /= Double(queries.count)
        #expect(recall10 >= 0.95)
    }

    @Test("IVF-flat recall@10 ≥ 0.95 on the benchmark sin/cos dim-384 generator")
    func benchLikeDim384Recall() throws {
        let dim = 384
        let count = 2_048
        let rows = benchLikeVectors(count: count, dim: dim, seedOffset: 42)
        let queries = benchLikeVectors(count: 16, dim: dim, seedOffset: 1_000_042)
        let buffer = try VectorBuffer(capacity: count, dim: dim)
        try buffer.batchInsert(vectors: rows, startingAt: 0)
        buffer.setCount(count)
        IVFFlatSearch.prepare(vectors: buffer, metric: .cosine, nlist: 64)
        var recall10 = 0.0
        for query in queries {
            let results = try IVFFlatSearch.search(
                query: query,
                vectors: buffer,
                k: 10,
                nlist: 64,
                nprobe: IVFFlatSearch.defaultNProbe,
                metric: .cosine
            )
            let predicted = results.map(\.internalID)
            let truth = bruteForceTopK(query: query, vectors: rows, k: 10, metric: .cosine)
            recall10 += recall(predicted: predicted, truth: truth, k: 10)
        }
        recall10 /= Double(queries.count)
        #expect(recall10 >= 0.95)
    }

    @Test("GraphIndex .fast uses shipped IVF-flat and keeps recall@10 ≥ 0.95")
    func graphIndexFastModeRecall() async throws {
        let dim = 64
        let count = 512
        let centers = seededVectors(count: 16, dim: dim, seed: 5)
        let rows = seededClusteredVectors(count: count, dim: dim, centers: centers, seed: 21)
        let ids = (0..<count).map { "v_\($0)" }
        let queries = seededClusteredVectors(count: 12, dim: dim, centers: centers, seed: 77)

        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 16,
                metric: .cosine,
                efSearch: 32,
                searchMode: .fast,
                ivfListCount: 32,
                ivfNProbe: IVFFlatSearch.defaultNProbe
            )
        )
        try await index.build(vectors: rows, ids: ids)

        var recall10 = 0.0
        for query in queries {
            let results = try await index.search(query: query, k: 10)
            let predicted = results.map(\.internalID)
            let truth = bruteForceTopK(query: query, vectors: rows, k: 10, metric: .cosine)
            recall10 += recall(predicted: predicted, truth: truth, k: 10)
        }
        recall10 /= Double(queries.count)
        #expect(recall10 >= 0.95)
    }

    @Test("Default GraphIndex search stays exact vs brute force")
    func defaultModeStaysExact() async throws {
        let dim = 32
        let count = 256
        let rows = seededVectors(count: count, dim: dim, seed: 9)
        let ids = (0..<count).map { "v_\($0)" }
        let query = seededVector(dim: dim, seed: 123)
        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 16, metric: .cosine, efSearch: 32)
        )
        try await index.build(vectors: rows, ids: ids)
        let results = try await index.search(query: query, k: 10)
        let predicted = results.map(\.internalID)
        let truth = bruteForceTopK(query: query, vectors: rows, k: 10, metric: .cosine)
        #expect(Set(predicted) == Set(truth))
    }

    @Test("nprobe >= nlist matches brute-force IDs")
    func fullProbeMatchesBruteForce() throws {
        let dim = 32
        let count = 256
        let k = 10
        let nlist = 16
        let rows = seededVectors(count: count, dim: dim, seed: 19)
        let query = seededVector(dim: dim, seed: 41)
        let buffer = try VectorBuffer(capacity: count, dim: dim)
        try buffer.batchInsert(vectors: rows, startingAt: 0)
        buffer.setCount(count)

        let results = try IVFFlatSearch.search(
            query: query,
            vectors: buffer,
            k: k,
            nlist: nlist,
            nprobe: nlist,
            metric: .cosine
        )
        let predicted = results.map(\.internalID)
        let truth = bruteForceTopK(query: query, vectors: rows, k: k, metric: .cosine)
        #expect(predicted == truth)
    }

    @Test("Wrong query dimension throws dimensionMismatch")
    func searchRejectsDimensionMismatch() throws {
        let rows = seededVectors(count: 256, dim: 16, seed: 2)
        let buffer = try VectorBuffer(capacity: 256, dim: 16)
        try buffer.batchInsert(vectors: rows, startingAt: 0)
        buffer.setCount(256)

        do {
            _ = try IVFFlatSearch.search(
                query: Array(repeating: 0, count: 15),
                vectors: buffer,
                k: 5,
                nlist: 16,
                nprobe: 4,
                metric: .cosine
            )
            Issue.record("Expected dimensionMismatch")
        } catch {
            guard case ANNSError.dimensionMismatch(let expected, let got) = error else {
                Issue.record("Expected dimensionMismatch, got \(error)")
                return
            }
            #expect(expected == 16)
            #expect(got == 15)
        }
    }

    @Test("GraphIndex .fast batchSearch throws on a mismatched query")
    func graphIndexFastBatchSearchRejectsDimensionMismatch() async throws {
        let dim = 32
        let count = 256
        let rows = seededVectors(count: count, dim: dim, seed: 8)
        let ids = (0..<count).map { "v_\($0)" }
        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 16,
                metric: .cosine,
                efSearch: 32,
                searchMode: .fast,
                ivfListCount: 16,
                ivfNProbe: 4
            )
        )
        try await index.build(vectors: rows, ids: ids)

        do {
            _ = try await index.batchSearch(
                queries: [Array(repeating: Float(0), count: dim - 1)],
                k: 5
            )
            Issue.record("Expected dimensionMismatch")
        } catch {
            guard case ANNSError.dimensionMismatch(let expected, let got) = error else {
                Issue.record("Expected dimensionMismatch, got \(error)")
                return
            }
            #expect(expected == dim)
            #expect(got == dim - 1)
        }
    }

    @Test("Insert after .fast build is visible to the next search")
    func insertInvalidatesCachedPartition() async throws {
        let dim = 32
        let count = 300
        var rows = seededClusteredVectors(
            count: count,
            dim: dim,
            centers: seededVectors(count: 12, dim: dim, seed: 4),
            seed: 13
        )
        let ids = (0..<count).map { "v_\($0)" }
        let index = GraphIndex(
            configuration: IndexConfiguration(
                degree: 16,
                metric: .cosine,
                efSearch: 32,
                searchMode: .fast,
                ivfListCount: 32,
                ivfNProbe: 8
            )
        )
        try await index.build(vectors: rows, ids: ids)

        var outlier = [Float](repeating: 0, count: dim)
        outlier[0] = 50
        try await index.insert(outlier, id: "outlier")
        rows.append(outlier)

        let results = try await index.search(query: outlier, k: 1)
        let first = try #require(results.first)
        #expect(first.id == "outlier")
    }
}
