import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Search Execution Engine Tests")
struct SearchExecutionTests {

    @Test("First eligible tier wins and later tiers plus ladder are skipped")
    func firstEligibleTierWins() async throws {
        let log = ExecutionLog()
        let request = try makeRequest()

        let output = try await GraphSearchEngine.run(
            request,
            tiers: [
                StubTier("flat", behavior: .results([stubResult(0)]), log: log),
                StubTier("hybrid", behavior: .results([stubResult(1)]), log: log),
            ],
            prepareCPU: {
                await log.append("ladder")
                return try makeLadderInputs()
            },
            metrics: nil
        )

        #expect(output.map(\.id) == ["r0"])
        #expect(await log.entries == ["flat"])
    }

    @Test("Generic tier failure degrades to the next tier without touching the ladder")
    func genericFailureFallsThroughToNextTier() async throws {
        let log = ExecutionLog()
        let request = try makeRequest()

        let output = try await GraphSearchEngine.run(
            request,
            tiers: [
                StubTier("flat", behavior: .fail("flat exploded"), log: log),
                StubTier("hybrid", behavior: .results([stubResult(2)]), log: log),
            ],
            prepareCPU: {
                try makeLadderInputs()
            },
            metrics: nil
        )

        #expect(output.map(\.id) == ["r2"])
        #expect(await log.entries == ["flat", "hybrid"])
    }

    @Test("CancellationError rethrows immediately without fallback or later tiers")
    func cancellationRethrowsImmediately() async throws {
        let log = ExecutionLog()
        let request = try makeRequest()

        do {
            _ = try await GraphSearchEngine.run(
                request,
                tiers: [
                    StubTier("flat", behavior: .cancel, log: log),
                    StubTier("hybrid", behavior: .results([stubResult(1)]), log: log),
                ],
                prepareCPU: {
                    await log.append("ladder")
                    return try makeLadderInputs()
                },
                metrics: nil
            )
            Issue.record("Expected CancellationError to propagate from the failing tier")
        } catch {
            #expect(error is CancellationError)
        }

        #expect(await log.entries == ["flat"])
    }

    @Test("Skip outcome continues the cascade through later tiers into the ladder once")
    func skipOutcomeContinuesCascadeIntoLadderOnce() async throws {
        let log = ExecutionLog()
        let request = try makeRequest()

        let output = try await GraphSearchEngine.run(
            request,
            tiers: [
                StubTier("flat", behavior: .skip, log: log),
                StubTier("hybrid", behavior: .fail("hybrid exploded"), log: log),
            ],
            prepareCPU: {
                await log.append("ladder")
                return PreparedCPUSearch(
                    vectors: [[1, 0, 0, 0]],
                    graph: [[(UInt32.max, 0)]],
                    hnsw: HNSWLayers(),
                    baseMetric: .l2
                )
            },
            metrics: nil
        )

        #expect(output.map(\.id) == ["r0"])
        #expect(await log.entries == ["flat", "hybrid", "ladder"])
    }

    @Test("CPU ladder cancellation propagates unwrapped")
    func ladderCancellationPropagatesUnwrapped() async throws {
        let log = ExecutionLog()
        let request = try makeRequest()

        do {
            _ = try await GraphSearchEngine.run(
                request,
                tiers: [
                    StubTier("flat", eligible: false, behavior: .results([stubResult(0)]), log: log)
                ],
                prepareCPU: {
                    await log.append("ladder")
                    throw CancellationError()
                },
                metrics: nil
            )
            Issue.record("Expected CancellationError to propagate from the CPU ladder")
        } catch {
            #expect(error is CancellationError)
        }

        #expect(await log.entries == ["ladder"])
    }

    @Test("Hybrid GPU gating keeps verbatim workload thresholds")
    func hybridGatingKeepsVerbatimThresholds() throws {
        let vectors = try VectorBuffer(capacity: 4_096, dim: 4)
        vectors.setCount(4_096)

        #expect(
            !HybridGPUBeamSearchTier.shouldUseHybridGPU(
                vectors: vectors, metric: .hamming, k: 256, ef: 256, degree: 64))
        #expect(
            !HybridGPUBeamSearchTier.shouldUseHybridGPU(
                vectors: vectors, metric: .l2, k: 256, ef: 256, degree: 63))
        #expect(
            HybridGPUBeamSearchTier.shouldUseHybridGPU(
                vectors: vectors, metric: .l2, k: 256, ef: 256, degree: 64))
        #expect(
            !HybridGPUBeamSearchTier.shouldUseHybridGPU(
                vectors: vectors, metric: .l2, k: 257, ef: 257, degree: 64))
    }

    @Test("Post-processing applies deletion, range cut, unmapped drop, and numeric mapping once")
    func postProcessingChainAppliesEveryStepOnce() throws {
        var softDeletion = SoftDeletion()
        softDeletion.markDeleted(2)
        let request = try makeRequest(softDeletion: softDeletion, maxDistance: 0.85)

        let raw = [
            SearchResult(id: "raw-a", score: 0.9, internalID: 0),
            SearchResult(id: "raw-b", score: 0.8, internalID: 1, numericID: 77),
            SearchResult(id: "raw-c", score: 0.5, internalID: 2),
            SearchResult(id: "raw-d", score: 0.3, internalID: 3),
        ]

        let output = GraphSearchEngine.postProcess(
            raw,
            softDeletion: request.softDeletion,
            filter: nil,
            metadataStore: request.metadataStore,
            idMap: request.idMap,
            maxDistance: request.maxDistance,
            limit: request.resultLimit
        )

        #expect(output.count == 1)
        #expect(output[0].id == "")
        #expect(output[0].score == 0.8)
        #expect(output[0].internalID == 1)
        #expect(output[0].numericID == 77)
    }

    @Test("Metadata filter step keeps only matching rows")
    func metadataFilterStepKeepsMatchingRows() throws {
        var metadataStore = MetadataStore()
        metadataStore.set("tag", stringValue: "keep", for: 1)
        let filter = _LegacySearchFilter.equals(column: "tag", value: "keep")
        let request = try makeRequest(metadataStore: metadataStore)

        let raw = [
            SearchResult(id: "raw-a", score: 0.9, internalID: 0),
            SearchResult(id: "raw-b", score: 0.8, internalID: 1, numericID: 77),
            SearchResult(id: "raw-c", score: 0.5, internalID: 2),
        ]

        let output = GraphSearchEngine.postProcess(
            raw,
            softDeletion: request.softDeletion,
            filter: filter,
            metadataStore: request.metadataStore,
            idMap: request.idMap,
            maxDistance: nil,
            limit: request.resultLimit
        )

        #expect(output.map(\.id) == [""])
        #expect(output.map(\.internalID) == [1])
    }

    @Test("Prefix trims mapped survivors to the requested limit")
    func prefixTrimsMappedSurvivorsToLimit() throws {
        let request = try makeRequest(limit: 2)
        let raw = [
            SearchResult(id: "raw-a", score: 0.3, internalID: 0),
            SearchResult(id: "raw-b", score: 0.4, internalID: 1),
            SearchResult(id: "raw-c", score: 0.5, internalID: 2),
            SearchResult(id: "raw-d", score: 0.6, internalID: 3),
        ]

        let output = GraphSearchEngine.postProcess(
            raw,
            softDeletion: request.softDeletion,
            filter: nil,
            metadataStore: request.metadataStore,
            idMap: request.idMap,
            maxDistance: nil,
            limit: request.resultLimit
        )

        #expect(output.map(\.id) == ["r0", ""])
    }

    // MARK: - Fixtures

    private func makeRequest(
        softDeletion: SoftDeletion = SoftDeletion(),
        metadataStore: MetadataStore = MetadataStore(),
        maxDistance: Float? = nil,
        limit: Int = 10
    ) throws -> SearchRequest {
        var idMap = IDMap()
        #expect(idMap.assign(externalID: "r0") == 0)
        #expect(idMap.assign(numericID: 77) == 1)
        #expect(idMap.assign(externalID: "r2") == 2)

        return SearchRequest(
            query: [0, 0, 0, 0],
            resultLimit: limit,
            metric: .l2,
            filter: nil,
            maxDistance: maxDistance,
            fetchK: 1,
            fetchEf: 4,
            context: nil,
            vectors: try VectorBuffer(capacity: 4, dim: 4),
            graph: try GraphBuffer(capacity: 4, degree: 2),
            entryPoint: 0,
            degree: 2,
            exactSearchMaxVectorCount: 1_024,
            baseMetric: .l2,
            softDeletion: softDeletion,
            metadataStore: metadataStore,
            idMap: idMap
        )
    }

    private func makeLadderInputs() throws -> PreparedCPUSearch {
        PreparedCPUSearch(
            vectors: [[1, 0, 0, 0]],
            graph: [[(UInt32.max, 0)]],
            hnsw: HNSWLayers(),
            baseMetric: .l2
        )
    }

    private func stubResult(_ internalID: UInt32) -> SearchResult {
        SearchResult(
            id: "stub-\(internalID)",
            score: Float(internalID) * 0.1 + 0.05,
            internalID: internalID
        )
    }

    private actor ExecutionLog {
        private(set) var entries: [String] = []

        func append(_ entry: String) {
            entries.append(entry)
        }
    }

    private struct StubTier: SearchTier {
        enum Behavior: Sendable {
            case results([SearchResult])
            case fail(String)
            case cancel
            case skip
        }

        let label: String
        let eligible: Bool
        let behavior: Behavior
        let log: ExecutionLog?

        init(_ label: String, eligible: Bool = true, behavior: Behavior, log: ExecutionLog? = nil) {
            self.label = label
            self.eligible = eligible
            self.behavior = behavior
            self.log = log
        }

        func isEligible(_ request: SearchRequest) -> Bool {
            eligible
        }

        func execute(_ request: SearchRequest) async throws -> [SearchResult]? {
            await log?.append(label)
            switch behavior {
            case .results(let results):
                return results
            case .skip:
                return nil
            case .cancel:
                throw CancellationError()
            case .fail(let message):
                throw ANNSError.searchFailed(message)
            }
        }
    }
}
