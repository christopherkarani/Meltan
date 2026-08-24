import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Batch Execution Fan-Out Tests")
struct BatchExecutionTests {

    // MARK: - Helper contract

    @Test("Results correlate back to input indexes despite out-of-order completion")
    func orderingPreservedUnderConcurrency() async throws {
        let count = 48
        var rng = SeededGenerator(state: 20_260_823)
        let expected = (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }

        let outputs = try await BatchExecution.run(
            over: Array(0..<count),
            maxConcurrency: 4
        ) { index in
            // Later indexes sleep less, so they finish first and force the
            // completion order to diverge from the input order.
            try await Task.sleep(for: .nanoseconds(Int64((count - index) * 200_000)))
            return expected[index]
        }

        #expect(outputs == expected)
    }

    @Test("First child error propagates unchanged and cancels in-flight siblings")
    func firstChildErrorPropagatesAndCancelsSiblings() async throws {
        let log = InvocationLog()

        do {
            _ = try await BatchExecution.run(
                over: Array(0..<16),
                maxConcurrency: 8
            ) { index in
                if index == 0 {
                    throw ProbeError()
                }
                do {
                    try await Task.sleep(for: .seconds(30))
                } catch is CancellationError {
                    await log.recordCancellation(index)
                }
                return index
            }
            Issue.record("Expected ProbeError to propagate from the batch")
        } catch {
            #expect(error is ProbeError)
        }

        #expect(await !log.cancellations.isEmpty)
    }

    @Test("When several children fail, one child error propagates instead of partial results")
    func multipleFailuresPropagateOneError() async throws {
        do {
            _ = try await BatchExecution.run(
                over: Array(0..<8),
                maxConcurrency: 8
            ) { index in
                throw index.isMultiple(of: 2) ? ProbeError() : AlternateProbeError()
            }
            Issue.record("Expected a child error to propagate from the batch")
        } catch {
            #expect(error is ProbeError || error is AlternateProbeError)
        }
    }

    @Test("Empty input yields empty output without running any child")
    func emptyInputYieldsEmptyOutput() async throws {
        let log = InvocationLog()

        let outputs = try await BatchExecution.run(
            over: [Int](),
            maxConcurrency: 4
        ) { index in
            await log.recordInvocation(index)
            return index
        }

        #expect(outputs.isEmpty)
        #expect(await log.invocations.isEmpty)
    }

    @Test("Single-element input maps exactly once and keeps its position")
    func singleElementInputMapsOnce() async throws {
        let log = InvocationLog()

        let outputs = try await BatchExecution.run(
            over: [7],
            maxConcurrency: 4
        ) { index in
            await log.recordInvocation(index)
            return index * 3
        }

        #expect(outputs == [21])
        #expect(await log.invocations == [7])
    }

    @Test("Non-positive maxConcurrency clamps to serial execution")
    func nonPositiveMaxConcurrencyClampsToSerial() async throws {
        let outputs = try await BatchExecution.run(
            over: [1, 2, 3],
            maxConcurrency: 0
        ) { $0 * 10 }

        #expect(outputs == [10, 20, 30])
    }

    // MARK: - GraphIndex.batchSearch adapter

    @Test("GraphIndex.batchSearch keeps every result correlated to its query index")
    func graphIndexBatchOrderingCorrelatesToQueryIndex() async throws {
        let index = try await makeGraphIndex()
        let vectors = makeVectors(count: 600, dim: 16, seedOffset: 0)

        var queries = Array(vectors.prefix(64))
        queries.append(vectors[599])

        var sequential: [[SearchResult]] = []
        sequential.reserveCapacity(queries.count)
        for query in queries {
            sequential.append(try await index.search(query: query, k: 5))
        }

        let results = try await index.batchSearch(queries: queries, k: 5)

        #expect(results.count == queries.count)
        for i in results.indices {
            // Positional correlation: slot i must answer query i, never another.
            #expect(Set(results[i].map(\.id)) == Set(sequential[i].map(\.id)))
        }
    }

    @Test("GraphIndex.batchSearch fails fast on the first failing query instead of masking it")
    func graphIndexBatchSearchFailsFast() async throws {
        let index = try await makeGraphIndex()

        var queries = makeVectors(count: 24, dim: 16, seedOffset: 300)
        queries[12] = Array(repeating: 0, count: 15)

        do {
            _ = try await index.batchSearch(queries: queries, k: 5)
            Issue.record("Expected the failing query to abort the whole batch")
        } catch {
            assertDimensionMismatch(error)
        }
    }

    @Test("GraphIndex.batchSearch handles empty and single-query batches")
    func graphIndexBatchSearchEdges() async throws {
        let index = try await makeGraphIndex()
        let vectors = makeVectors(count: 600, dim: 16, seedOffset: 0)

        #expect(try await index.batchSearch(queries: [], k: 5).isEmpty)

        let singleResult = try await index.batchSearch(queries: [vectors[7]], k: 5)
        let direct = try await index.search(query: vectors[7], k: 5)
        #expect(singleResult.count == 1)
        #expect(Set(singleResult[0].map(\.id)) == Set(direct.map(\.id)))
    }

    // MARK: - ShardedIndex.batchSearch adapter

    @Test("ShardedIndex.batchSearch keeps every result correlated to its query index")
    func shardedIndexBatchOrderingCorrelatesToQueryIndex() async throws {
        let vectors = makeClusteredVectors(count: 1500, dim: 32, clusters: 12)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let index = Advanced.ShardedIndex(
            numShards: 4,
            nprobe: 3,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )
        try await index.build(vectors: vectors, ids: ids)

        let queries = Array(vectors.prefix(100))
        let results = try await index.batchSearch(queries: queries, k: 10)

        #expect(results.count == queries.count)
        var hits = 0
        for i in results.indices where results[i].contains(where: { $0.id == "v\(i)" }) {
            hits += 1
        }
        let recall = Float(hits) / Float(results.count)
        #expect(recall > 0.55)
    }

    @Test("ShardedIndex.batchSearch fails fast on the first failing query instead of masking it")
    func shardedIndexBatchSearchFailsFast() async throws {
        let vectors = makeClusteredVectors(count: 600, dim: 32, clusters: 12)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let index = Advanced.ShardedIndex(
            numShards: 4,
            nprobe: 3,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )
        try await index.build(vectors: vectors, ids: ids)

        var queries = Array(vectors.prefix(24))
        queries[12] = Array(repeating: 0, count: 31)

        do {
            _ = try await index.batchSearch(queries: queries, k: 10)
            Issue.record("Expected the failing query to abort the whole batch")
        } catch {
            assertDimensionMismatch(error)
        }
    }

    @Test("ShardedIndex.batchSearch handles empty and single-query batches")
    func shardedIndexBatchSearchEdges() async throws {
        let vectors = makeClusteredVectors(count: 600, dim: 32, clusters: 12)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let index = Advanced.ShardedIndex(
            numShards: 4,
            nprobe: 3,
            configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96)
        )
        try await index.build(vectors: vectors, ids: ids)

        #expect(try await index.batchSearch(queries: [], k: 10).isEmpty)

        let singleResult = try await index.batchSearch(queries: [vectors[7]], k: 10)
        #expect(singleResult.count == 1)
        #expect(singleResult[0].contains(where: { $0.id == "v7" }))

        // k above the fused exact-scan cap must not clamp per-shard fetch
        // below the caller k (self-match still has to survive the merge).
        let deepK = FlatGPUSearch.maxTopK + 44
        let deepResult = try await index.batchSearch(queries: [vectors[7]], k: deepK)
        #expect(deepResult.count == 1)
        #expect(deepResult[0].count > 10)
        #expect(deepResult[0].count <= deepK)
        #expect(deepResult[0].contains(where: { $0.id == "v7" }))
    }

    // MARK: - Fixtures

    private func makeGraphIndex() async throws -> GraphIndex {
        let vectors = makeVectors(count: 600, dim: 16, seedOffset: 0)
        let ids = (0..<vectors.count).map { "v\($0)" }
        let index = GraphIndex(
            configuration: IndexConfiguration(degree: 8, metric: .cosine),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)
        return index
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    private func makeClusteredVectors(count: Int, dim: Int, clusters: Int) -> [[Float]] {
        (0..<count).map { i in
            let cluster = i % clusters
            return (0..<dim).map { d in
                let center: Float = d == (cluster % dim) ? 1.0 : 0.0
                let noiseSeed = Float((i * dim) + d)
                return center + sin(noiseSeed * 0.031) * 0.01
            }
        }
    }

    private func assertDimensionMismatch(_ error: Error) {
        if case ANNSError.dimensionMismatch = error {
            return
        }
        Issue.record("Expected dimensionMismatch, got \(error)")
    }

    private struct ProbeError: Error, Sendable {}

    private struct AlternateProbeError: Error, Sendable {}

    private actor InvocationLog {
        private(set) var invocations: [Int] = []
        private(set) var cancellations: [Int] = []

        func recordInvocation(_ index: Int) {
            invocations.append(index)
        }

        func recordCancellation(_ index: Int) {
            cancellations.append(index)
        }
    }
}
