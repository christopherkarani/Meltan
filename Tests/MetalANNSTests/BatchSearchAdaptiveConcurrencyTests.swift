import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Batch Search Adaptive Concurrency Tests")
struct BatchSearchAdaptiveConcurrencyTests {
    @Test("gpuModeUsesQueuePoolCount")
    func gpuModeUsesQueuePoolCount() async throws {
        #if targetEnvironment(simulator)
        return
        #else
        guard let context = makeContextOrSkip() else {
            return
        }

        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 8, metric: .cosine),
            context: context
        )

        let maxConcurrency = await index.batchSearchMaxConcurrencyForTesting()
        let queueCount = await context.queuePool.queues.count
        #expect(maxConcurrency == queueCount)
        #endif
    }

    @Test("cpuModeUsesProcessorCount")
    func cpuModeUsesProcessorCount() async throws {
        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 8, metric: .cosine),
            context: nil
        )

        let maxConcurrency = await index.batchSearchMaxConcurrencyForTesting()
        #expect(maxConcurrency == max(1, ProcessInfo.processInfo.activeProcessorCount))
    }

    @Test("batchSearchResultsUnchanged")
    func batchSearchResultsUnchanged() async throws {
        let vectors = makeVectors(count: 600, dim: 32, seedOffset: 0)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let queries = Array(vectors.prefix(100))
        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 8, metric: .cosine),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        var sequential: [[SearchResult]] = []
        sequential.reserveCapacity(queries.count)
        for query in queries {
            sequential.append(try await index.search(query: query, k: 10))
        }

        let batch = try await index.batchSearch(queries: queries, k: 10)
        #expect(batch.count == sequential.count)
        for i in 0..<batch.count {
            #expect(Set(batch[i].map { $0.id }) == Set(sequential[i].map { $0.id }))
        }
    }

    private func makeContextOrSkip() -> MetalContext? {
        do {
            return try MetalContext()
        } catch {
            return nil
        }
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }
}
