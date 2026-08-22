import Foundation
import Metal
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

/// Executor-level coverage for the vector-search seam: every adapter is
/// reachable through `GraphIndex.search(path:)`, and strict paths refuse
/// rather than silently substitute.
@Suite("VectorSearch Paths")
struct VectorSearchPathTests {
    private func makeGPUContextOrSkip() -> MetalContext? {
        #if targetEnvironment(simulator)
        print("Skipping VectorSearchPathTests on simulator")
        return nil
        #else
        guard MTLCreateSystemDefaultDevice() != nil else {
            print("Skipping VectorSearchPathTests GPU cases: no Metal device available")
            return nil
        }
        do {
            return try MetalContext()
        } catch {
            print("Skipping VectorSearchPathTests GPU cases: MetalContext unavailable (\(error))")
            return nil
        }
        #endif
    }

    private func makeIndex(
        vectors: [[Float]],
        context: MetalContext?,
        configuration: IndexConfiguration = IndexConfiguration(degree: 8, efSearch: 32)
    ) async throws -> GraphIndex {
        let index = GraphIndex(configuration: configuration, context: context)
        let ids = (0..<vectors.count).map { "v_\($0)" }
        try await index.build(vectors: vectors, ids: ids)
        return index
    }

    @Test func gpuPathIsStrictWithoutMetalContext() async throws {
        // An index constructed without a Metal context cannot serve .gpu,
        // even on machines that have a GPU — strictness is per-index.
        let vectors = seededVectors(count: 128, dim: 16)
        let index = try await makeIndex(vectors: vectors, context: nil)
        await #expect(throws: ANNSError.self) {
            try await index.search(query: seededVector(dim: 16), k: 5, path: .gpu)
        }
    }

    @Test func gpuPathRejectsWorkloadsAboveBeamCap() async throws {
        // effectiveEf = max(efSearch, effectiveK) exceeds the 256 beam cap;
        // the cap guard fires before the context guard, so this is
        // deterministic with or without Metal.
        var configuration = IndexConfiguration.default
        configuration.efSearch = 300
        let vectors = seededVectors(count: 300, dim: 8)
        let index = try await makeIndex(vectors: vectors, context: nil, configuration: configuration)
        await #expect(throws: ANNSError.self) {
            try await index.search(query: seededVector(dim: 8), k: 10, path: .gpu)
        }
    }

    @Test func cpuPathServesQueriesWithoutMetal() async throws {
        let vectors = seededVectors(count: 256, dim: 16)
        let index = try await makeIndex(vectors: vectors, context: nil)
        let results = try await index.search(query: seededVector(dim: 16), k: 10, path: .cpu)
        #expect(results.count == 10)
        #expect(Set(results.map(\.id)).count == 10)
    }

    @Test func exactPathMatchesBruteForceReference() async throws {
        let dim = 12
        let vectors = seededVectors(count: 200, dim: dim)
        let query = seededVector(dim: dim, seed: 7)

        let reference = vectors.enumerated()
            .map { (index: $0.offset, distance: SIMDDistance.distance(query, $0.element, metric: .cosine)) }
            .sorted { $0.distance < $1.distance }
        let expectedTop3 = Set(reference.prefix(3).map { "v_\($0.index)" })
        let expectedTop5 = Set(reference.prefix(5).map { "v_\($0.index)" })

        // Context-free on purpose: the exact tier's host scan serves small
        // corpora without a GPU dispatch.
        let index = try await makeIndex(vectors: vectors, context: nil)
        let results = try await index.search(query: query, k: 5, path: .exact)

        #expect(results.count == 5)
        let returnedIDs = Set(results.map(\.id))
        #expect(returnedIDs.intersection(expectedTop3) == expectedTop3)
        #expect(returnedIDs.intersection(expectedTop5).count >= 4)
    }

    @Test func exactPathRefusesWhenDeletionsExist() async throws {
        let vectors = seededVectors(count: 100, dim: 8)
        let index = try await makeIndex(vectors: vectors, context: nil)
        try await index.delete(id: "v_0")

        // Exact scans ignore deletion filtering, so the strict path must
        // refuse instead of returning deleted rows.
        await #expect(throws: ANNSError.self) {
            try await index.search(query: seededVector(dim: 8), k: 5, path: .exact)
        }
    }

    @Test func autoPathStillServesSmallCorpora() async throws {
        let vectors = seededVectors(count: 128, dim: 16)
        let index = try await makeIndex(vectors: vectors, context: nil)
        let results = try await index.search(query: seededVector(dim: 16), k: 5, path: .auto)
        #expect(results.count == 5)
    }

    @Test func gpuAndCPUPathsAgreeOnSeededData() async throws {
        guard let context = makeGPUContextOrSkip() else {
            return
        }

        let dim = 32
        let vectors = seededVectors(count: 512, dim: dim)
        let index = try await makeIndex(vectors: vectors, context: context)

        for seed in 0..<10 {
            let query = seededVector(dim: dim, seed: UInt64(100 + seed))
            let gpuResults = try await index.search(query: query, k: 10, path: .gpu)
            let cpuResults = try await index.search(query: query, k: 10, path: .cpu)

            #expect(gpuResults.count == 10)
            #expect(cpuResults.count == 10)

            let gpuIDs = Set(gpuResults.map(\.id))
            let cpuIDs = Set(cpuResults.map(\.id))
            let overlap = Double(gpuIDs.intersection(cpuIDs).count) / 10.0
            #expect(overlap >= 0.5, "seed \(seed): GPU/CPU overlap \(overlap)")
        }
    }
}
