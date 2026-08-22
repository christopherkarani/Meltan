import Foundation
import MetalANNSCore
import Testing

@testable import MetalANNS

@Suite("Advanced.StreamingIndex Merge Tests")
struct StreamingIndexMergeTests {
    @Test("Merge preserves all vectors")
    func mergePreservesAllVectors() async throws {
        let deltaCapacity = 10
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: deltaCapacity,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<(deltaCapacity + 5)).map { makeVector(row: $0, dim: 8) }
        for (i, vector) in vectors.enumerated() {
            try await index.insert(vector, id: "v\(i)")
        }

        #expect(await index.count == deltaCapacity + 5)

        for (i, vector) in vectors.enumerated() {
            let results = try await index.search(query: vector, k: 1)
            #expect(!results.isEmpty)
            #expect(results[0].id == "v\(i)")
        }
    }

    @Test("Background merge triggered")
    func backgroundMergeTriggered() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 8,
                mergeStrategy: .background
            ))

        for i in 0..<20 {
            try await index.insert(makeVector(row: i, dim: 8), id: "v\(i)")
        }

        try await index.flush()

        #expect(await index.count == 20)
        #expect(await index.isMerging == false)
    }

    @Test("Merge clears isMerging")
    func mergeClearsIsMerging() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 500,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<300).map { makeVector(row: $0, dim: 16) }
        let ids = (0..<300).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        let flushTask = Task {
            try await index.flush()
        }

        var sawMerging = false
        for _ in 0..<400 {
            if await index.isMerging {
                sawMerging = true
                break
            }
            try await Task.sleep(nanoseconds: 2_000_000)
        }

        try await flushTask.value

        #expect(sawMerging)
        #expect(await index.isMerging == false)
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.091) + cos(i * 0.037)
        }
    }
}

@Suite("Streaming Merge Planner Tests")
struct StreamingMergePlannerTests {
    private func makeVector(_ seed: Float, dim: Int = 2) -> [Float] {
        [seed, seed + 1]
    }

    private func makeLedger(
        pendingVectors: [[Float]] = [],
        pendingIDs: [String] = [],
        historyIDs: [String] = [],
        historyVectors: [[Float]] = [],
        deletedIDs: Set<String> = [],
        vectorDimension: Int? = nil,
        hasDelta: Bool = false
    ) -> StreamingLedger {
        StreamingLedger(
            pendingVectors: pendingVectors,
            pendingIDs: pendingIDs,
            historyVectorData: historyVectors.flatMap { $0 },
            historyIDs: historyIDs,
            deletedIDs: deletedIDs,
            vectorDimension: vectorDimension ?? (historyVectors.first?.count),
            hasDelta: hasDelta
        )
    }

    @Test("Delta overflow appends when a delta exists")
    func deltaOverflowAppends() {
        let ledger = makeLedger(
            pendingVectors: [makeVector(1), makeVector(2), makeVector(3)],
            pendingIDs: ["a", "b", "c"],
            hasDelta: true
        )

        let action = StreamingMergePlanner.planDeltaOverflow(ledger)
        let expected = StreamingRecordSet(
            vectors: [makeVector(1), makeVector(2), makeVector(3)],
            ids: ["a", "b", "c"]
        )

        #expect(action == .appendToDelta(expected))
    }

    @Test("Delta overflow rebuilds delta from pending records")
    func deltaOverflowRebuildsDelta() {
        let ledger = makeLedger(
            pendingVectors: [makeVector(1), makeVector(2)],
            pendingIDs: ["a", "b"]
        )

        let action = StreamingMergePlanner.planDeltaOverflow(ledger)

        #expect(action == .rebuildDelta(StreamingRecordSet(vectors: [makeVector(1), makeVector(2)], ids: ["a", "b"])))
    }

    @Test("Delta overflow idles on empty or lone pending records")
    func deltaOverflowIdles() {
        let emptyPending = makeLedger(hasDelta: true)
        let lonePendingNoDelta = makeLedger(pendingVectors: [makeVector(1)], pendingIDs: ["a"])

        #expect(StreamingMergePlanner.planDeltaOverflow(emptyPending) == .idle)
        #expect(StreamingMergePlanner.planDeltaOverflow(lonePendingNoDelta) == .idle)
    }

    @Test("Capacity overflow recovery follows merge state and error type")
    func overflowRecoveryPlans() {
        let capacityError = ANNSError.capacityExceeded(capacity: 8)

        #expect(
            StreamingMergePlanner.planOverflowRecovery(for: capacityError, isMerging: false) == .triggerMerge)
        #expect(
            StreamingMergePlanner.planOverflowRecovery(for: capacityError, isMerging: true)
                == .waitForCurrentMerge)
        #expect(
            StreamingMergePlanner.planOverflowRecovery(for: ANNSError.constructionFailed("other"), isMerging: false)
                == .rethrow)
    }

    @Test("Merge with empty base and prefix resets to empty")
    func mergeEmptyResets() {
        let ledger = makeLedger(historyIDs: [], historyVectors: [])

        let plan = StreamingMergePlanner.planMerge(
            baseSnapshotCount: 0,
            baseRecords: .empty,
            prefix: ledger.activeRecords(upperBound: 0),
            tailLedger: ledger
        )

        #expect(plan == .resetToEmpty)
    }

    @Test("Merge with single merged record rebuilds delta from retained records")
    func mergeSingleRecordRebuildsDelta() {
        let ledger = makeLedger(historyIDs: ["only"], historyVectors: [makeVector(7)])

        let plan = StreamingMergePlanner.planMerge(
            baseSnapshotCount: 1,
            baseRecords: .empty,
            prefix: ledger.activeRecords(upperBound: 1),
            tailLedger: ledger
        )

        guard case .rebuildDelta(let retained) = plan else {
            Issue.record("Expected rebuildDelta plan, got \(plan)")
            return
        }

        #expect(retained == StreamingRecordSet(vectors: [makeVector(7)], ids: ["only"]))
        #expect(StreamingMergePlanner.planTailDisposition(retained) == .parkPending(retained))
    }

    @Test("Merge replaces base with snapshot prefix and plans concurrent-appends tail separately")
    func mergeWithConcurrentAppends() {
        // History grew past the snapshot while the base snapshot was fetched:
        // prefix is bounded by the pre-await count, the tail lives beyond it.
        // Base records predate the history window and never overlap it.
        let ledger = makeLedger(
            historyIDs: ["h-0", "h-1", "tail-0", "tail-1"],
            historyVectors: [makeVector(11), makeVector(12), makeVector(13), makeVector(14)]
        )
        let prefix = ledger.activeRecords(upperBound: 2)

        let plan = StreamingMergePlanner.planMerge(
            baseSnapshotCount: 2,
            baseRecords: StreamingRecordSet(vectors: [makeVector(10)], ids: ["old-0"]),
            prefix: prefix,
            tailLedger: ledger
        )

        guard case .replaceBase(let newBaseRecords) = plan else {
            Issue.record("Expected replaceBase plan, got \(plan)")
            return
        }

        #expect(
            newBaseRecords
                == StreamingRecordSet(
                    vectors: [makeVector(10), makeVector(11), makeVector(12)],
                    ids: ["old-0", "h-0", "h-1"]
                ))

        let tail = ledger.activeRecords(in: 2..<ledger.historyIDs.count)
        #expect(tail == StreamingRecordSet(vectors: [makeVector(13), makeVector(14)], ids: ["tail-0", "tail-1"]))
        #expect(StreamingMergePlanner.planTailDisposition(tail) == .rebuildDelta(tail))
    }

    @Test("Planners exclude deleted IDs from prefix and tail slices")
    func deletedIDExclusion() {
        let dim = 2
        let ledger = makeLedger(
            historyIDs: ["kept-0", "gone", "kept-1", "tail"],
            historyVectors: [makeVector(20), makeVector(99), makeVector(21), makeVector(22)],
            deletedIDs: ["gone"],
            vectorDimension: dim
        )

        let prefix = ledger.activeRecords(upperBound: 3)
        #expect(prefix.ids == ["kept-0", "kept-1"])
        #expect(prefix.vectors == [makeVector(20), makeVector(21)])

        let tail = ledger.activeRecords(in: 3..<ledger.historyIDs.count)
        #expect(tail.ids == ["tail"])
        #expect(tail.vectors == [makeVector(22)])
        #expect(StreamingMergePlanner.planTailDisposition(tail) == .parkPending(tail))
    }

    @Test("Tail disposition covers discard, park, and rebuild outcomes")
    func tailDispositions() {
        #expect(StreamingMergePlanner.planTailDisposition(.empty) == .discard)
        #expect(
            StreamingMergePlanner.planTailDisposition(StreamingRecordSet(vectors: [makeVector(1)], ids: ["a"]))
                == .parkPending(StreamingRecordSet(vectors: [makeVector(1)], ids: ["a"])))
        #expect(
            StreamingMergePlanner.planTailDisposition(
                StreamingRecordSet(vectors: [makeVector(1), makeVector(2)], ids: ["a", "b"]))
                == .rebuildDelta(StreamingRecordSet(vectors: [makeVector(1), makeVector(2)], ids: ["a", "b"])))
    }

    @Test("Shared filter ladder matches store rows across all operators")
    func sharedFilterLadderMatchesStoreRows() throws {
        var store = MetadataStore()
        store.set("genre", stringValue: "rock", for: 0)
        store.set("year", floatValue: 1998, for: 0)
        store.set("plays", intValue: 42, for: 0)
        store.set("year", intValue: 2001, for: 1)

        #expect(store.matches(id: 0, filter: .equals(column: "genre", value: "rock")))
        #expect(!store.matches(id: 1, filter: .equals(column: "genre", value: "rock")))
        #expect(!store.matches(id: 2, filter: .equals(column: "genre", value: "rock")))
        #expect(store.matches(id: 0, filter: .greaterThan(column: "year", value: 1997)))
        #expect(store.matches(id: 0, filter: .lessThan(column: "year", value: 1999)))
        #expect(!store.matches(id: 0, filter: .greaterThan(column: "year", value: 1998)))
        #expect(store.matches(id: 1, filter: .greaterThan(column: "year", value: 1999)))
        #expect(store.matches(id: 0, filter: .lessThanInt(column: "plays", value: 100)))
        #expect(!store.matches(id: 1, filter: .lessThanInt(column: "plays", value: 100)))
        #expect(store.matches(id: 0, filter: .in(column: "genre", values: ["rock", "pop"])))
        // id 0: genre=rock, year=1998.0, plays=42
        #expect(
            !store.matches(
                id: 0,
                filter: .and([
                    .equals(column: "genre", value: "rock"),
                    .or([.greaterThanInt(column: "plays", value: 50), .not(.lessThan(column: "year", value: 1999))]),
                ])))
        #expect(
            store.matches(
                id: 0,
                filter: .and([
                    .equals(column: "genre", value: "rock"),
                    .or([.greaterThanInt(column: "plays", value: 50), .lessThan(column: "year", value: 1999)]),
                ])))
    }
}
