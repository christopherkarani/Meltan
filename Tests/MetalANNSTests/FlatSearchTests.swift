import Metal
import Testing
@testable import MetalANNSCore

@Suite("Flat Exact GPU Search Tests")
struct FlatSearchTests {
    static func makeContext() throws -> MetalContext {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        return try MetalContext()
    }

    func seededVectors(count: Int, dim: Int, seed: UInt64 = 7) -> [[Float]] {
        var state = seed
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2.0 - 1.0
        }
        return (0..<count).map { _ in (0..<dim).map { _ in next() } }
    }

    func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> [(id: UInt32, distance: Float)] {
        let scored = vectors.enumerated().map { index, vector -> (UInt32, Float) in
            (UInt32(index), distance(query: query, vector: vector, metric: metric))
        }
        return scored
            .sorted { lhs, rhs in
                lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
            }
            .prefix(k)
            .map { ($0.0, $0.1) }
    }

    func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
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
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
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

    @Test("Exact recall across metrics and sizes")
    func parityAcrossMetricsAndSizes() async throws {
        let context = try Self.makeContext()
        let dim = 128

        for vectorCount in [64, 100, 2048, 2049, 5000] {
            let vectors = seededVectors(count: vectorCount, dim: dim, seed: UInt64(vectorCount))
            let buffer = try VectorBuffer(capacity: vectorCount, dim: dim, device: context.device)
            try buffer.batchInsert(vectors: vectors, startingAt: 0)
            buffer.setCount(vectorCount)

            for metric in [Metric.cosine, .l2, .innerProduct] {
                let query = seededVectors(count: 1, dim: dim, seed: UInt64(vectorCount) &+ 999)[0]
                let results = try await FlatGPUSearch.search(
                    context: context,
                    query: query,
                    vectors: buffer,
                    k: 100,
                    metric: metric
                )
                let expected = bruteForceTopK(query: query, vectors: vectors, k: 100, metric: metric)

                #expect(results.count == min(100, vectorCount), "result count mismatch n=\(vectorCount) metric=\(metric)")
                let gotIDs = Set(results.map(\.internalID))
                let wantIDs = Set(expected.map(\.id))
                #expect(gotIDs == wantIDs, "top-k mismatch n=\(vectorCount) metric=\(metric)")

                // Scores must be ascending and match the expected distances.
                for (index, result) in results.enumerated() where index > 0 {
                    #expect(result.score >= results[index - 1].score - 1e-6,
                            "scores not ascending n=\(vectorCount) metric=\(metric)")
                }
                if let first = results.first, let bestExpected = expected.first {
                    #expect(abs(first.score - bestExpected.distance) < 1e-3,
                            "best score mismatch n=\(vectorCount) metric=\(metric): \(first.score) vs \(bestExpected.distance)")
                }
            }
        }
    }

    @Test("Batch search matches per-query brute force")
    func batchParity() async throws {
        let context = try Self.makeContext()
        let dim = 128
        let vectorCount = 4200
        let vectors = seededVectors(count: vectorCount, dim: dim, seed: 31)
        let buffer = try VectorBuffer(capacity: vectorCount, dim: dim, device: context.device)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(vectorCount)

        let queries = seededVectors(count: 16, dim: dim, seed: 77)
        let batches = try await FlatGPUSearch.batchSearch(
            context: context,
            queries: queries,
            vectors: buffer,
            k: 50,
            metric: .cosine
        )

        #expect(batches.count == queries.count)
        for (queryIndex, batch) in batches.enumerated() {
            let expected = bruteForceTopK(
                query: queries[queryIndex],
                vectors: vectors,
                k: 50,
                metric: .cosine
            )
            #expect(Set(batch.map(\.internalID)) == Set(expected.map(\.id)),
                    "batch top-k mismatch for query \(queryIndex)")
        }
    }

    @Test("k greater than corpus size returns all vectors")
    func kExceedsCorpus() async throws {
        let context = try Self.makeContext()
        let dim = 32
        let vectorCount = 70
        let vectors = seededVectors(count: vectorCount, dim: dim)
        let buffer = try VectorBuffer(capacity: vectorCount, dim: dim, device: context.device)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(vectorCount)

        let query = seededVectors(count: 1, dim: dim, seed: 5)[0]
        let results = try await FlatGPUSearch.search(
            context: context,
            query: query,
            vectors: buffer,
            k: 256,
            metric: .l2
        )
        #expect(results.count == vectorCount)
    }

    @Test("Norm cache invalidation stays exact after in-place mutation")
    func normCacheInvalidation() async throws {
        let context = try Self.makeContext()
        let dim = 64
        let vectorCount = 512
        let vectors = seededVectors(count: vectorCount, dim: dim, seed: 13)
        let buffer = try VectorBuffer(capacity: vectorCount, dim: dim, device: context.device)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(vectorCount)

        let query = seededVectors(count: 1, dim: dim, seed: 21)[0]

        // Prime the norm cache.
        _ = try await FlatGPUSearch.search(context: context, query: query, vectors: buffer, k: 20, metric: .cosine)
        _ = try await FlatGPUSearch.search(context: context, query: query, vectors: buffer, k: 20, metric: .l2)

        // Mutate several rows in place (count unchanged).
        var mutated = vectors
        for row in [3, 100, 411] {
            let replacement = seededVectors(count: 1, dim: dim, seed: UInt64(row) &+ 500)[0]
            try buffer.insert(vector: replacement, at: row)
            mutated[row] = replacement
        }

        for metric in [Metric.cosine, .l2] {
            let results = try await FlatGPUSearch.search(
                context: context, query: query, vectors: buffer, k: 25, metric: metric
            )
            let expected = bruteForceTopK(query: query, vectors: mutated, k: 25, metric: metric)
            #expect(Set(results.map(\.internalID)) == Set(expected.map(\.id)),
                    "stale norm cache after mutation for metric \(metric)")
        }

        FlatGPUSearch.invalidateHostNormCache()
    }

    @Test("Eligibility gate excludes unsupported storage")
    func eligibilityGate() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }
        let context = try Self.makeContext()
        let smallBuffer = try VectorBuffer(capacity: 8, dim: 16, device: context.device)
        #expect(!FlatGPUSearch.isEligible(vectors: smallBuffer, metric: .cosine, k: 10, maxVectorCount: 1_000_000))

        let goodBuffer = try VectorBuffer(capacity: 500, dim: 16, device: context.device)
        goodBuffer.setCount(500)
        #expect(FlatGPUSearch.isEligible(vectors: goodBuffer, metric: .cosine, k: 10, maxVectorCount: 1_000_000))
        #expect(FlatGPUSearch.isEligible(vectors: goodBuffer, metric: .l2, k: 256, maxVectorCount: 1_000_000))
        #expect(!FlatGPUSearch.isEligible(vectors: goodBuffer, metric: .hamming, k: 10, maxVectorCount: 1_000_000))
        #expect(!FlatGPUSearch.isEligible(vectors: goodBuffer, metric: .cosine, k: 257, maxVectorCount: 1_000_000))
        #expect(!FlatGPUSearch.isEligible(vectors: goodBuffer, metric: .cosine, k: 10, maxVectorCount: 0))
        #expect(!FlatGPUSearch.isEligible(vectors: goodBuffer, metric: .cosine, k: 10, maxVectorCount: 100))
    }
}
