import Accelerate
import Foundation
import Metal
import Testing

@testable import MetalANNSCore

/// Regression suite proving the residual-bound exact cascade returns
/// brute-force EXACT results across metrics, dimensions (incl. non-multiple-
/// of-4 tails), adversarial distributions (duplicates, zero norms),
/// cache invalidation after in-place writes, and deterministic reruns.
@Suite("Residual Cascade Tests")
struct ResidualCascadeTests {
    // MARK: - Helpers

    private func makeContext() throws -> MetalContext {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        return try MetalContext()
    }

    private struct SeededGenerator {
        var state: UInt64
        init(seed: UInt64) {
            state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 0xD1B5_4A32_D192_ED03
        }
        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 33) & 0xFFFF_FFFF) / Float(0xFFFF_FFFF) * 2.0 - 1.0
        }
    }

    private func makeVectors(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = SeededGenerator(seed: seed)
        return (0..<count).map { _ in (0..<dim).map { _ in rng.next() } }
    }

    /// Scalar reference implementation — deliberately independent of the
    /// library's vectorized kernels.
    private func referenceDistance(query: [Float], vector: [Float], metric: Metric) -> Double {
        switch metric {
        case .cosine:
            var dot = 0.0
            var normQ = 0.0
            var normV = 0.0
            for dimension in 0..<query.count {
                let qValue = Double(query[dimension])
                let vValue = Double(vector[dimension])
                dot += qValue * vValue
                normQ += qValue * qValue
                normV += vValue * vValue
            }
            let denom = (normQ * normV).squareRoot()
            return denom < 1e-10 ? 1.0 : 1.0 - dot / denom
        case .l2:
            var sum = 0.0
            for dimension in 0..<query.count {
                let diff = Double(query[dimension]) - Double(vector[dimension])
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot = 0.0
            for dimension in 0..<query.count {
                dot += Double(query[dimension]) * Double(vector[dimension])
            }
            return -dot
        case .hamming:
            return .infinity
        }
    }

    /// Sorted ascending (id, distance) reference list.
    private func referenceRanking(
        query: [Float], vectors: [[Float]], metric: Metric
    ) -> [(id: UInt32, distance: Double)] {
        var scored = vectors.enumerated().map { index, vector -> (UInt32, Double) in
            (UInt32(index), referenceDistance(query: query, vector: vector, metric: metric))
        }
        scored.sort { lhs, rhs in
            lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
        }
        return scored.map { (id: $0.0, distance: $0.1) }
    }

    private func makeBuffer(_ vectors: [[Float]], dim: Int) throws -> VectorBuffer {
        let buffer = try VectorBuffer(capacity: vectors.count, dim: dim)
        try buffer.batchInsert(vectors: vectors, startingAt: 0)
        buffer.setCount(vectors.count)
        return buffer
    }

    /// Asserts the returned results are the brute-force top-k. When the
    /// k-boundary sits inside a float near-tie group, id-set equality is
    /// replaced by membership in every valid top-k set.
    private func assertExact(
        actual: [SearchResult],
        reference: [(id: UInt32, distance: Double)],
        neighborTotal: Int
    ) {
        let effectiveK = min(neighborTotal, FlatGPUSearch.maxTopK, reference.count)
        let boundaryDistance = reference[effectiveK - 1].distance
        let gapAtK =
            reference.count > effectiveK
            ? reference[effectiveK].distance - boundaryDistance
            : Double.infinity

        let actualIDs = Set(actual.prefix(effectiveK).map(\.internalID))
        #expect(actualIDs.count == effectiveK, "duplicate ids in results")

        if gapAtK > 2e-3 {
            let expectedIDs = Set(reference.prefix(effectiveK).map(\.id))
            #expect(actualIDs == expectedIDs, "top-k id set mismatch")
        } else {
            var valid = Set<UInt32>()
            for entry in reference where entry.distance <= boundaryDistance + 2e-3 {
                valid.insert(entry.id)
            }
            #expect(
                actualIDs.isSubset(of: valid),
                "results contain rows outside the tied top-k window"
            )
        }

        let referenceByID = Dictionary(uniqueKeysWithValues: reference.map { ($0.id, $0.distance) })
        for result in actual.prefix(effectiveK) {
            guard let expected = referenceByID[result.internalID] else {
                Issue.record("returned id \(result.internalID) missing from corpus ranking")
                continue
            }
            let tolerance = max(2e-3, abs(expected) * 2e-3)
            #expect(
                abs(Double(result.score) - expected) <= tolerance,
                "score mismatch id=\(result.internalID) got=\(result.score) want=\(expected)"
            )
        }

        let slice = Array(actual.prefix(effectiveK))
        for index in 1..<slice.count {
            #expect(
                slice[index - 1].score <= slice[index].score,
                "results not ascending at \(index)"
            )
        }
    }

    private let corpusSize = ResidualCascade.minVectorCount + 512

    // MARK: - Fuzz exactness

    @Test("Residual cascade is brute-force exact across metrics, dims, seeds")
    func exactAcrossMetricsDimsAndSeeds() async throws {
        let context = try makeContext()
        let dims = [384, 333, 16]
        let metrics: [Metric] = [.cosine, .l2, .innerProduct]
        let ks = [1, 24, 97]

        for dim in dims {
            for seed in [UInt64(7), UInt64(99)] {
                let vectors = makeVectors(count: corpusSize, dim: dim, seed: seed)
                let buffer = try makeBuffer(vectors, dim: dim)
                var rng = SeededGenerator(seed: seed &+ 123)
                for queryIndex in 0..<2 {
                    let query = (0..<dim).map { _ in rng.next() }
                    for metric in metrics {
                        let ranking = referenceRanking(query: query, vectors: vectors, metric: metric)
                        for neighborTotal in ks {
                            let results = try await runSearch(
                                context: context, query: query, buffer: buffer,
                                neighborTotal: neighborTotal, metric: metric
                            )
                            assertExact(
                                actual: results, reference: ranking, neighborTotal: neighborTotal
                            )
                        }
                    }
                }
            }
        }
    }

    @Test("Duplicate-heavy corpora stay exact under tie pressure")
    func duplicatesStayExact() async throws {
        let context = try makeContext()
        let dim = 16
        let baseVectors = makeVectors(count: corpusSize / 8, dim: dim, seed: 5150)
        var vectors = [Float]()
        vectors.reserveCapacity(corpusSize * dim)
        var flattened = [[Float]]()
        flattened.reserveCapacity(corpusSize)
        for rowIndex in 0..<corpusSize {
            let source = baseVectors[rowIndex % baseVectors.count]
            flattened.append(source)
        }
        _ = vectors
        let buffer = try makeBuffer(flattened, dim: dim)

        var rng = SeededGenerator(seed: 777)
        let metrics: [Metric] = [.cosine, .l2, .innerProduct]
        for metric in metrics {
            let query = (0..<dim).map { _ in rng.next() }
            let ranking = referenceRanking(query: query, vectors: flattened, metric: metric)
            let results = try await runSearch(
                context: context, query: query, buffer: buffer, neighborTotal: 50, metric: metric
            )
            assertExact(actual: results, reference: ranking, neighborTotal: 50)
        }
    }

    @Test("Zero-norm rows and zero-norm cosine queries stay exact")
    func zeroNormEdgeCases() async throws {
        let context = try makeContext()
        let dim = 16
        var vectors = makeVectors(count: corpusSize, dim: dim, seed: 8080)
        // Plant zero rows at scattered indices.
        for zeroIndex in stride(from: 100, to: corpusSize, by: 40_000) {
            vectors[zeroIndex] = [Float](repeating: 0, count: dim)
        }
        let buffer = try makeBuffer(vectors, dim: dim)

        // Zero query: every distance must be defined as 1.0, ids lowest-first.
        let zeroQuery = [Float](repeating: 0, count: dim)
        let zeroResults = try await runSearch(
            context: context, query: zeroQuery, buffer: buffer, neighborTotal: 10, metric: .cosine
        )
        #expect(zeroResults.count == 10)
        for (index, result) in zeroResults.enumerated() {
            #expect(result.internalID == UInt32(index))
            #expect(result.score == 1.0)
        }

        // Regular query against a corpus containing zero rows.
        var rng = SeededGenerator(seed: 9090)
        let query = (0..<dim).map { _ in rng.next() }
        for metric in [.cosine, .l2, .innerProduct] as [Metric] {
            let ranking = referenceRanking(query: query, vectors: vectors, metric: metric)
            let results = try await runSearch(
                context: context, query: query, buffer: buffer, neighborTotal: 33, metric: metric
            )
            assertExact(actual: results, reference: ranking, neighborTotal: 33)
        }
    }

    @Test("In-place insert invalidates cached bounds and stays exact")
    func invalidationOnInsert() async throws {
        let context = try makeContext()
        let dim = 24
        let vectors = makeVectors(count: corpusSize, dim: dim, seed: 2024)
        let buffer = try makeBuffer(vectors, dim: dim)

        var rng = SeededGenerator(seed: 3131)
        let query = (0..<dim).map { _ in rng.next() }
        _ = try await runSearch(context: context, query: query, buffer: buffer, neighborTotal: 20, metric: .cosine)

        // Overwrite rows with fresh vectors; caches must not serve stale bounds.
        var mutated = vectors
        var replacementRng = SeededGenerator(seed: 4242)
        let replacedIndices = [5, 123_456, corpusSize - 1]
        for target in replacedIndices {
            let replacement = (0..<dim).map { _ in replacementRng.next() }
            try buffer.insert(vector: replacement, at: target)
            mutated[target] = replacement
        }

        let ranking = referenceRanking(query: query, vectors: mutated, metric: .cosine)
        let results = try await runSearch(
            context: context, query: query, buffer: buffer, neighborTotal: 20, metric: .cosine
        )
        assertExact(actual: results, reference: ranking, neighborTotal: 20)
    }

    @Test("Repeated searches are deterministic")
    func determinism() async throws {
        let context = try makeContext()
        let dim = 16
        let vectors = makeVectors(count: corpusSize, dim: dim, seed: 4711)
        let buffer = try makeBuffer(vectors, dim: dim)
        var rng = SeededGenerator(seed: 1212)
        let query = (0..<dim).map { _ in rng.next() }

        let first = try await runSearch(context: context, query: query, buffer: buffer, neighborTotal: 24, metric: .cosine)
        let second = try await runSearch(context: context, query: query, buffer: buffer, neighborTotal: 24, metric: .cosine)
        #expect(first.map { $0.internalID } == second.map { $0.internalID })
        #expect(first.map { $0.score } == second.map { $0.score })
    }

    // MARK: - Entry points

    private func runSearch(
        context: MetalContext,
        query: [Float],
        buffer: VectorBuffer,
        neighborTotal: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        // Route through the production entry point so tier selection,
        // eligibility gates, and cache wiring are exercised too.
        return try await FlatGPUSearch.search(
            context: context, query: query, vectors: buffer, k: neighborTotal, metric: metric
        )
    }
}
