import Foundation
import MetalANNSCore

enum StreamingMetadataValue: Sendable, Codable {
    case string(String)
    case float(Float)
    case int64(Int64)
}

/// An ordered set of vector records exchanged between the streaming index and
/// its planners.
struct StreamingRecordSet: Sendable, Equatable {
    var vectors: [[Float]]
    var ids: [String]

    static let empty = StreamingRecordSet(vectors: [], ids: [])

    init(vectors: [[Float]], ids: [String]) {
        self.vectors = vectors
        self.ids = ids
    }

    var count: Int {
        ids.count
    }

    func appending(_ other: StreamingRecordSet) -> StreamingRecordSet {
        var vectors = self.vectors
        vectors.append(contentsOf: other.vectors)
        var ids = self.ids
        ids.append(contentsOf: other.ids)
        return StreamingRecordSet(vectors: vectors, ids: ids)
    }
}

/// A value snapshot of `_StreamingIndex` state captured at a decision point.
///
/// The actor captures ledger values synchronously between await points and
/// hands them to `StreamingMergePlanner`. Planners read only these values —
/// never live actor state — so every merge decision is reproducible in tests
/// without an actor or a Metal device.
struct StreamingLedger: Sendable {
    var pendingVectors: [[Float]]
    var pendingIDs: [String]
    /// Flat history payload; record `i` occupies `historyVectorData[i * dim..< (i + 1) * dim]`.
    var historyVectorData: [Float]
    var historyIDs: [String]
    var allIDs: Set<String>
    var deletedIDs: Set<String>
    var idInBase: Set<String>
    var idInDelta: Set<String>
    var metadataByID: [String: [String: StreamingMetadataValue]]
    var vectorDimension: Int?
    var hasDelta: Bool

    init(
        pendingVectors: [[Float]] = [],
        pendingIDs: [String] = [],
        historyVectorData: [Float] = [],
        historyIDs: [String] = [],
        allIDs: Set<String> = [],
        deletedIDs: Set<String> = [],
        idInBase: Set<String> = [],
        idInDelta: Set<String> = [],
        metadataByID: [String: [String: StreamingMetadataValue]] = [:],
        vectorDimension: Int? = nil,
        hasDelta: Bool = false
    ) {
        self.pendingVectors = pendingVectors
        self.pendingIDs = pendingIDs
        self.historyVectorData = historyVectorData
        self.historyIDs = historyIDs
        self.allIDs = allIDs
        self.deletedIDs = deletedIDs
        self.idInBase = idInBase
        self.idInDelta = idInDelta
        self.metadataByID = metadataByID
        self.vectorDimension = vectorDimension
        self.hasDelta = hasDelta
    }

    func vector(atLogicalIndex index: Int) -> [Float] {
        guard let dim = vectorDimension else {
            return []
        }
        let start = index * dim
        let end = start + dim
        if start < 0 || end > historyVectorData.count {
            return []
        }
        return Array(historyVectorData[start..<end])
    }

    /// Active (non-deleted) records among the first `upperBound` history slots.
    func activeRecords(upperBound: Int) -> StreamingRecordSet {
        let safeUpper = min(upperBound, historyIDs.count)
        guard safeUpper > 0 else {
            return .empty
        }

        var vectors: [[Float]] = []
        var ids: [String] = []
        vectors.reserveCapacity(safeUpper)
        ids.reserveCapacity(safeUpper)

        for index in 0..<safeUpper {
            let id = historyIDs[index]
            guard !deletedIDs.contains(id) else {
                continue
            }
            ids.append(id)
            vectors.append(vector(atLogicalIndex: index))
        }
        return StreamingRecordSet(vectors: vectors, ids: ids)
    }

    /// Active (non-deleted) records within a history range.
    func activeRecords(in range: Range<Int>) -> StreamingRecordSet {
        let lower = max(0, range.lowerBound)
        let upper = min(historyIDs.count, range.upperBound)
        guard lower < upper else {
            return .empty
        }

        var vectors: [[Float]] = []
        var ids: [String] = []
        vectors.reserveCapacity(upper - lower)
        ids.reserveCapacity(upper - lower)

        for index in lower..<upper {
            let id = historyIDs[index]
            guard !deletedIDs.contains(id) else {
                continue
            }
            ids.append(id)
            vectors.append(vector(atLogicalIndex: index))
        }
        return StreamingRecordSet(vectors: vectors, ids: ids)
    }
}

enum StreamingDeltaAction: Sendable, Equatable {
    /// Stop the flush loop (nothing actionable right now).
    case idle
    /// No delta exists yet; build one from the given pending records.
    case rebuildDelta(StreamingRecordSet)
    /// A delta exists; append the given pending records to it.
    case appendToDelta(StreamingRecordSet)
}

enum StreamingOverflowRecovery: Sendable, Equatable {
    case rethrow
    case waitForCurrentMerge
    case triggerMerge
}

enum StreamingMergePlan: Sendable, Equatable {
    /// Nothing to merge anywhere; reset the streaming containers to empty.
    case resetToEmpty
    /// Fewer than two merged records exist; drop base/delta and rebuild the
    /// delta from the retained records (parking a lone record as pending).
    case rebuildDelta(records: StreamingRecordSet)
    /// Build a new base from the merged snapshot-prefix records. Tail records
    /// appended while the base rebuilds are planned separately afterwards,
    /// mirroring the original post-build tail read.
    case replaceBase(newBaseRecords: StreamingRecordSet)
}

enum StreamingTailDisposition: Sendable, Equatable {
    case discard
    case parkPending(StreamingRecordSet)
    case rebuildDelta(StreamingRecordSet)
}

/// Pure decision functions over `StreamingLedger` values.
///
/// These reproduce the exact control flow of `_StreamingIndex`'s former inline
/// flush loop and `triggerMerge` body. The actor executes the returned plans;
/// no function here reads or mutates actor state.
enum StreamingMergePlanner {
    /// Decides the next action for the pending-flush loop.
    static func planDeltaOverflow(_ ledger: StreamingLedger) -> StreamingDeltaAction {
        guard !ledger.pendingVectors.isEmpty else {
            return .idle
        }
        guard ledger.hasDelta else {
            guard ledger.pendingVectors.count >= 2 else {
                return .idle
            }
            return .rebuildDelta(
                StreamingRecordSet(vectors: ledger.pendingVectors, ids: ledger.pendingIDs)
            )
        }
        return .appendToDelta(
            StreamingRecordSet(vectors: ledger.pendingVectors, ids: ledger.pendingIDs)
        )
    }

    /// Decides how to react to a failed delta append. `isMerging` must be read
    /// by the actor at recovery time (after the failed await), not captured
    /// earlier.
    static func planOverflowRecovery(for error: ANNSError, isMerging: Bool) -> StreamingOverflowRecovery {
        guard case .capacityExceeded = error else {
            return .rethrow
        }
        return isMerging ? .waitForCurrentMerge : .triggerMerge
    }

    /// Decides the merge branch from the base snapshot and pre-await history
    /// prefix. `prefix` must be sliced from a ledger captured before the base
    /// snapshot fetch so concurrent deletes cannot shift it across the await;
    /// `tailLedger` is captured after that fetch, which is where the original
    /// implementation read tails for the sub-two-record branch.
    static func planMerge(
        baseSnapshotCount: Int,
        baseRecords: StreamingRecordSet,
        prefix: StreamingRecordSet,
        tailLedger: StreamingLedger
    ) -> StreamingMergePlan {
        let merged = baseRecords.appending(prefix)

        guard !merged.ids.isEmpty else {
            return .resetToEmpty
        }

        guard merged.ids.count >= 2 else {
            let tail = tailLedger.activeRecords(in: baseSnapshotCount..<tailLedger.historyIDs.count)
            return .rebuildDelta(records: merged.appending(tail))
        }

        return .replaceBase(newBaseRecords: merged)
    }

    /// Decides what happens to post-snapshot tail records once their count is
    /// known.
    static func planTailDisposition(_ records: StreamingRecordSet) -> StreamingTailDisposition {
        switch records.count {
        case 0:
            return .discard
        case 1:
            return .parkPending(records)
        default:
            return .rebuildDelta(records)
        }
    }
}
