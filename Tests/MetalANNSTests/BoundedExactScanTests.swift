import Accelerate
import Foundation
import Metal
import Testing

@testable import MetalANNSCore

/// Regression suite proving the int8-bounded prefilter returns brute-force
/// EXACT results across metrics, dimensions (incl. non-multiple-of-4 tails),
/// adversarial value distributions, candidate-budget growth, and in-place
/// mutations that invalidate cached quantized codes.
@Suite("Bounded Exact Scan Tests")
struct BoundedExactScanTests {
    // MARK: - Helpers

    private func makeContext() throws -> MetalContext {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw ANNSError.deviceNotSupported
        }
        return try MetalContext()
    }

    private struct SeededGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 0xD1B5_4A32_D192_ED03 }
        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 33) & 0xFFFF_FFFF) / Float(0xFFFF_FFFF) * 2.0 - 1.0
        }
        mutating func uniform01() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 40) & 0xFFFF_FF) / Float(0xFF_FFFF)
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
            for d in 0..<query.count {
                let q = Double(query[d])
                let v = Double(vector[d])
                dot += q * v
                normQ += q * q
                normV += v * v
            }
            let denom = (normQ * normV).squareRoot()
            return denom < 1e-10 ? 1.0 : 1.0 - dot / denom
        case .l2:
            var sum = 0.0
            for d in 0..<query.count {
                let diff = Double(query[d]) - Double(vector[d])
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot = 0.0
            for d in 0..<query.count {
                dot += Double(query[d]) * Double(vector[d])
            }
            return -dot
        case .hamming:
            return .infinity
        }
    }

    /// Sorted ascending (distance, id) reference list.
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

    /// Runs the bounded scan through the production entry point.
    private func runBounded(
        context: MetalContext,
        query: [Float],
        buffer: VectorBuffer,
        k: Int,
        metric: Metric
    ) -> [SearchResult]? {
        FlatGPUSearch.boundedExactSearch(
            query: query,
            vectors: buffer,
            k: k,
            metric: metric
        )
    }

    /// Asserts the returned results are the brute-force top-k. When the
    /// k-boundary sits inside a float near-tie (or duplicate) group, id-set
    /// equality is replaced by membership in every valid top-k set.
    private func assertExact(
        actual: [SearchResult],
        reference: [(id: UInt32, distance: Double)],
        k: Int
    ) {
        // The production entry point caps k at FlatGPUSearch.maxTopK.
        let effectiveK = min(k, FlatGPUSearch.maxTopK, reference.count)
        let boundaryDistance = reference[effectiveK - 1].1
        let gapAtK =
            reference.count > effectiveK
            ? reference[effectiveK].1 - boundaryDistance
            : Double.infinity

        let actualIDs = Set(actual.prefix(effectiveK).map(\.internalID))
        #expect(actualIDs.count == effectiveK, "duplicate ids in results")

        if gapAtK > 2e-3 {
            let expectedIDs = Set(reference.prefix(effectiveK).map(\.id))
            #expect(actualIDs == expectedIDs, "top-k id set mismatch")
        } else {
            // Any row within the tie window of the boundary may legitimately
            // appear; anything beyond it must not.
            var valid = Set<UInt32>()
            for entry in reference where entry.distance <= boundaryDistance + 2e-3 {
                valid.insert(entry.id)
            }
            #expect(
                actualIDs.isSubset(of: valid),
                "results contain rows outside the tied top-k window"
            )
        }

        // Every returned score must match its reference distance.
        let referenceByID = Dictionary(uniqueKeysWithValues: reference.map { ($0.id, $0.1) })
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

        // Ascending order invariant.
        let slice = Array(actual.prefix(effectiveK))
        for index in 1..<slice.count {
            #expect(
                slice[index - 1].score <= slice[index].score,
                "results not ascending at \(index)"
            )
        }
    }

    private let corpusSize = BoundedExactScan.minVectorCount + 512

    // MARK: - Fuzz exactness

    @Test("Bounded scan is brute-force exact across metrics, dims, seeds")
    func exactAcrossMetricsDimsAndSeeds() throws {
        let context = try makeContext()
        let dims = [384, 128, 333, 16]
        let metrics: [Metric] = [.cosine, .l2, .innerProduct]
        let ks = [1, 24, 97]

        for dim in dims {
            for seed in [UInt64(7), UInt64(99)] {
                let vectors = makeVectors(count: corpusSize, dim: dim, seed: seed)
                let buffer = try makeBuffer(vectors, dim: dim)
                var rng = SeededGenerator(seed: seed &+ 123)
                for queryIndex in 0..<4 {
                    let query = (0..<dim).map { _ in rng.next() }
                    for metric in metrics {
                        let ranking = referenceRanking(query: query, vectors: vectors, metric: metric)
                        for k in ks {
                            guard
                                let actual = runBounded(
                                    context: context, query: query, buffer: buffer,
                                    k: k, metric: metric
                                )
                            else {
                                Issue.record("boundedExactSearch returned nil (dim=\(dim))")
                                continue
                            }
                            assertExact(actual: actual, reference: ranking, k: k)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Adversarial distributions

    @Test("Adversarial distributions stay exact: zeros, duplicates, extremes")
    func adversarialDataStaysExact() throws {
        let context = try makeContext()
        for metric in [Metric.cosine, .l2, .innerProduct] {
            let dim = 64
            let count = corpusSize
            var rng = SeededGenerator(seed: 2024)
            var vectors = [[Float]](repeating: [], count: count)

            // Quarter zeros, quarter duplicates of row 0, rest random with
            // wild magnitude spread.
            let base = (0..<dim).map { _ in rng.next() }
            for row in 0..<count {
                switch row % 4 {
                case 0:
                    vectors[row] = [Float](repeating: 0, count: dim)
                case 1:
                    vectors[row] = base
                default:
                    let magnitude: Float = row % 8 == 0 ? 1e-5 : (row % 8 == 4 ? 1e5 : 1.0)
                    vectors[row] = (0..<dim).map { _ in rng.next() * magnitude }
                }
            }

            let buffer = try makeBuffer(vectors, dim: dim)
            let query = (0..<dim).map { _ in rng.next() }
            let ranking = referenceRanking(query: query, vectors: vectors, metric: metric)
            for k in [24, 256] {
                guard
                    let actual = runBounded(
                        context: context, query: query, buffer: buffer, k: k, metric: metric
                    )
                else {
                    Issue.record("boundedExactSearch returned nil for adversarial data (\(metric))")
                    continue
                }
                assertExact(actual: actual, reference: ranking, k: k)
            }
        }
    }

    @Test("All-zero query stays exact")
    func zeroQueryStaysExact() throws {
        let context = try makeContext()
        let dim = 96
        let vectors = makeVectors(count: corpusSize, dim: dim, seed: 31)
        let buffer = try makeBuffer(vectors, dim: dim)
        let query = [Float](repeating: 0, count: dim)

        for metric in [Metric.cosine, .l2, .innerProduct] {
            let ranking = referenceRanking(query: query, vectors: vectors, metric: metric)
            guard
                let actual = runBounded(
                    context: context, query: query, buffer: buffer, k: 24, metric: metric
                )
            else {
                Issue.record("boundedExactSearch returned nil for zero query")
                continue
            }
            assertExact(actual: actual, reference: ranking, k: 24)
        }
    }

    // MARK: - Candidate budget growth

    @Test("Large k forces multi-round growth and stays exact")
    func largeKMultiRoundGrowthStaysExact() throws {
        let context = try makeContext()
        let dim = 384
        let vectors = makeVectors(count: corpusSize, dim: dim, seed: 55)
        let buffer = try makeBuffer(vectors, dim: dim)
        var rng = SeededGenerator(seed: 77)
        let query = (0..<dim).map { _ in rng.next() }

        // k near the production cap (FlatGPUSearch.maxTopK = 256); large
        // candidate sets exercise deeper selection passes.
        for k in [97, 256] {
            for metric in [Metric.cosine, .l2, .innerProduct] {
                let ranking = referenceRanking(query: query, vectors: vectors, metric: metric)
                guard
                    let actual = runBounded(
                        context: context, query: query, buffer: buffer, k: k, metric: metric
                    )
                else {
                    Issue.record("boundedExactSearch returned nil for k=\(k)")
                    continue
                }
                assertExact(actual: actual, reference: ranking, k: k)
            }
        }
    }

    // MARK: - Cache invalidation

    @Test("Codes invalidate after in-place mutation")
    func invalidationAfterMutationStaysExact() throws {
        let context = try makeContext()
        let dim = 128
        let vectors = makeVectors(count: corpusSize, dim: dim, seed: 88)
        let buffer = try makeBuffer(vectors, dim: dim)
        var rng = SeededGenerator(seed: 89)
        let query = (0..<dim).map { _ in rng.next() }

        // Warm the code cache with one search.
        _ = runBounded(context: context, query: query, buffer: buffer, k: 24, metric: .l2)

        // Mutate a row in place.
        let replacement = (0..<dim).map { _ in rng.next() * 10 }
        try buffer.insert(vector: replacement, at: 0)
        var mutated = vectors
        mutated[0] = replacement

        for metric in [Metric.cosine, .l2, .innerProduct] {
            let ranking = referenceRanking(query: query, vectors: mutated, metric: metric)
            guard
                let actual = runBounded(
                    context: context, query: query, buffer: buffer, k: 24, metric: metric
                )
            else {
                Issue.record("boundedExactSearch returned nil after mutation")
                continue
            }
            assertExact(actual: actual, reference: ranking, k: 24)
        }
    }

    // MARK: - Eligibility gate

    @Test("Below threshold returns nil so callers fall back")
    func belowThresholdReturnsNil() throws {
        let context = try makeContext()
        let dim = 64
        let vectors = makeVectors(count: 2048, dim: dim, seed: 12)
        let buffer = try makeBuffer(vectors, dim: dim)
        let query = (0..<dim).map { _ in Float(0.25) }
        let actual = runBounded(
            context: context, query: query, buffer: buffer, k: 10, metric: .cosine
        )
        #expect(actual == nil)
    }

    // MARK: - Slice coverage

    @Test("Rows near the corpus tail are always scanned (slice coverage)")
    func sliceCoverageCoversEntireCorpus() throws {
        let context = try makeContext()
        let dim = 64
        // Size chosen so naive fixed-chunk slicing can under-cover.
        let count = 4 * BoundedExactScan.minVectorCount
        var vectors = makeVectors(count: count, dim: dim, seed: 404)
        let queriesAndTargets: [(query: [Float], targetRow: Int)] = [
            (vectors[count - 1], count - 1),
            (vectors[count - 7], count - 7),
            (vectors[count / 2 + 3], count / 2 + 3),
        ]
        // Perturb targets slightly so they are strict nearest neighbors.
        var rng = SeededGenerator(seed: 405)
        for index in 0..<queriesAndTargets.count {
            var base = vectors[queriesAndTargets[index].targetRow]
            for d in 0..<dim { base[d] += rng.next() * 1e-3 }
            vectors[queriesAndTargets[index].targetRow] = base
        }

        let buffer = try makeBuffer(vectors, dim: dim)
        for (rawQuery, targetRow) in queriesAndTargets {
            var query = rawQuery
            for d in 0..<dim { query[d] += rng.next() * 1e-3 }
            guard
                let actual = runBounded(
                    context: context, query: query, buffer: buffer, k: 5, metric: .l2
                )
            else {
                Issue.record("boundedExactSearch returned nil")
                continue
            }
            let ranking = referenceRanking(query: query, vectors: vectors, metric: .l2)
            #expect(
                ranking[0].id == UInt32(targetRow),
                "reference disagrees on target (got \(ranking[0].id), want \(targetRow))"
            )
            assertExact(actual: actual, reference: ranking, k: 5)
        }
    }
}
