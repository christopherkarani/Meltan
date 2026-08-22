import Foundation
import MetalANNSCore

/// A continuous-ingest index that uses a two-level merge architecture.
///
/// New vectors land in a small `delta` index. When the delta reaches
/// `StreamingConfiguration.deltaCapacity` it is merged into the frozen
/// `base` index asynchronously (or synchronously for `.blocking` strategy).
/// Search always probes both shards and also covers pre-build pending inserts.
public actor _StreamingIndex {
    private struct PersistedMeta: Sendable, Codable {
        let config: StreamingConfiguration
        let vectorDimension: Int?
        let allVectorData: [Float]
        let allIDsList: [String]
        let deletedIDs: [String]
        let metadataByID: [String: [String: StreamingMetadataValue]]

        private enum CodingKeys: String, CodingKey {
            case config
            case vectorDimension
            case allVectorData
            case allVectorsList
            case allIDsList
            case deletedIDs
            case metadataByID
        }

        init(
            config: StreamingConfiguration,
            vectorDimension: Int?,
            allVectorData: [Float],
            allIDsList: [String],
            deletedIDs: [String],
            metadataByID: [String: [String: StreamingMetadataValue]]
        ) {
            self.config = config
            self.vectorDimension = vectorDimension
            self.allVectorData = allVectorData
            self.allIDsList = allIDsList
            self.deletedIDs = deletedIDs
            self.metadataByID = metadataByID
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            config = try container.decode(StreamingConfiguration.self, forKey: .config)
            vectorDimension = try container.decodeIfPresent(Int.self, forKey: .vectorDimension)
            allIDsList = try container.decode([String].self, forKey: .allIDsList)
            deletedIDs = try container.decode([String].self, forKey: .deletedIDs)
            metadataByID = try container.decode([String: [String: StreamingMetadataValue]].self, forKey: .metadataByID)

            if let flat = try container.decodeIfPresent([Float].self, forKey: .allVectorData) {
                allVectorData = flat
            } else if let legacy = try container.decodeIfPresent([[Float]].self, forKey: .allVectorsList) {
                allVectorData = legacy.flatMap { $0 }
            } else {
                allVectorData = []
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(config, forKey: .config)
            try container.encodeIfPresent(vectorDimension, forKey: .vectorDimension)
            try container.encode(allVectorData, forKey: .allVectorData)
            try container.encode(allIDsList, forKey: .allIDsList)
            try container.encode(deletedIDs, forKey: .deletedIDs)
            try container.encode(metadataByID, forKey: .metadataByID)
        }
    }

    private var base: GraphIndex?
    private var delta: GraphIndex?
    private var mergeTask: Task<Void, Error>?
    private var lastBackgroundMergeError: ANNSError?
    private var _isMerging = false

    private var pendingVectors: [[Float]] = []
    private var pendingIDs: [String] = []

    private var allVectorData: [Float] = []
    private var allIDsList: [String] = []
    private var allIDs: Set<String> = []
    private var deletedIDs: Set<String> = []

    private var idInBase: Set<String> = []
    private var idInDelta: Set<String> = []
    private var metadataByID: [String: [String: StreamingMetadataValue]] = [:]
    private var vectorDimension: Int?

    private let config: StreamingConfiguration
    /// Optional metrics sink for streaming operations.
    /// When set, the same instance is propagated to child base/delta indexes so
    /// delegated searches/inserts are recorded exactly once at the child level.
    public var metrics: IndexMetrics? = nil {
        didSet { metricsNeedsPropagation = true }
    }
    private var metricsNeedsPropagation = false

    public init(config: StreamingConfiguration = .default) {
        self.config = config
    }

    public var count: Int {
        allIDs.count - deletedIDs.count
    }

    public var isMerging: Bool {
        _isMerging
    }

    public func setMetrics(_ metrics: IndexMetrics?) {
        self.metrics = metrics
    }

    public func insert(_ vector: [Float], id: String) async throws {
        try checkBackgroundMergeError()
        guard !allIDs.contains(id) else {
            throw ANNSError.idAlreadyExists(id)
        }
        try validateDimension(of: vector)
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        allIDs.insert(id)
        allIDsList.append(id)
        allVectorData.append(contentsOf: vector)
        pendingIDs.append(id)
        pendingVectors.append(vector)

        try await flushPendingIntoDelta()
        try await maybeTriggerMerge()
    }

    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
        try checkBackgroundMergeError()
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }

        var seen = Set<String>()
        for id in ids {
            if !seen.insert(id).inserted || allIDs.contains(id) {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        for vector in vectors {
            try validateDimension(of: vector)
        }
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        for id in ids {
            allIDs.insert(id)
            allIDsList.append(id)
        }
        for vector in vectors {
            allVectorData.append(contentsOf: vector)
        }
        pendingIDs.append(contentsOf: ids)
        pendingVectors.append(contentsOf: vectors)

        try await flushPendingIntoDelta()
        try await maybeTriggerMerge()
    }

    public func search(
        query: [Float],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        try checkBackgroundMergeError()
        try validateQueryDimension(query)
        guard k > 0 else {
            return []
        }
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        let searchMetric = metric ?? config.indexConfiguration.metric
        var combined: [SearchResult] = []
        combined.reserveCapacity(k * 2)

        if let base {
            let baseResults = try await base.search(query: query, k: k, filter: filter, metric: metric)
            combined.append(contentsOf: baseResults)
        }

        if let delta {
            let deltaResults = try await delta.search(query: query, k: k, filter: filter, metric: metric)
            combined.append(contentsOf: deltaResults)
        }

        combined.append(contentsOf: pendingSearchResults(query: query, filter: filter, metric: searchMetric))
        return Array(dedupeAndSort(combined).prefix(k))
    }

    public func batchSearch(
        queries: [[Float]],
        k: Int,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [[SearchResult]] {
        guard !queries.isEmpty else {
            return []
        }

        return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
            for (index, query) in queries.enumerated() {
                group.addTask { [self] in
                    let results = try await self.search(query: query, k: k, filter: filter, metric: metric)
                    return (index, results)
                }
            }

            var ordered = [[SearchResult]?](repeating: nil, count: queries.count)
            for try await (index, results) in group {
                ordered[index] = results
            }
            return ordered.map { $0 ?? [] }
        }
    }

    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        filter: _LegacySearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        try checkBackgroundMergeError()
        try validateQueryDimension(query)
        guard maxDistance >= 0 else {
            return []
        }
        guard limit > 0 else {
            return []
        }
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        let searchMetric = metric ?? config.indexConfiguration.metric
        var combined: [SearchResult] = []

        if let base {
            let baseResults = try await base.rangeSearch(
                query: query,
                maxDistance: maxDistance,
                limit: Int.max,
                filter: filter,
                metric: metric
            )
            combined.append(contentsOf: baseResults)
        }

        if let delta {
            let deltaResults = try await delta.rangeSearch(
                query: query,
                maxDistance: maxDistance,
                limit: Int.max,
                filter: filter,
                metric: metric
            )
            combined.append(contentsOf: deltaResults)
        }

        for result in pendingSearchResults(query: query, filter: filter, metric: searchMetric)
        where result.score <= maxDistance {
            combined.append(result)
        }

        return Array(dedupeAndSort(combined).prefix(limit))
    }

    public func setMetadata(_ column: String, value: String, for id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        var row = metadataByID[id] ?? [:]
        row[column] = .string(value)
        metadataByID[id] = row

        if idInBase.contains(id), let base {
            try await base.setMetadata(column, value: value, for: id)
        }
        if idInDelta.contains(id), let delta {
            try await delta.setMetadata(column, value: value, for: id)
        }
    }

    public func setMetadata(_ column: String, value: Float, for id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        var row = metadataByID[id] ?? [:]
        row[column] = .float(value)
        metadataByID[id] = row

        if idInBase.contains(id), let base {
            try await base.setMetadata(column, value: value, for: id)
        }
        if idInDelta.contains(id), let delta {
            try await delta.setMetadata(column, value: value, for: id)
        }
    }

    public func setMetadata(_ column: String, value: Int64, for id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        var row = metadataByID[id] ?? [:]
        row[column] = .int64(value)
        metadataByID[id] = row

        if idInBase.contains(id), let base {
            try await base.setMetadata(column, value: value, for: id)
        }
        if idInDelta.contains(id), let delta {
            try await delta.setMetadata(column, value: value, for: id)
        }
    }

    public func delete(id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        deletedIDs.insert(id)
        idInBase.remove(id)
        idInDelta.remove(id)
        metadataByID.removeValue(forKey: id)

        if let base {
            do {
                try await base.delete(id: id)
            } catch let error as ANNSError {
                guard case .idNotFound = error else {
                    throw error
                }
            }
        }

        if let delta {
            do {
                try await delta.delete(id: id)
            } catch let error as ANNSError {
                guard case .idNotFound = error else {
                    throw error
                }
            }
        }

        removePendingID(id)
        removeFromPendingHistory(id)
    }

    public func flush() async throws {
        try checkBackgroundMergeError()
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        if let task = mergeTask {
            defer { mergeTask = nil }
            try await task.value
        }

        try await flushPendingIntoDelta()

        if delta != nil || !pendingVectors.isEmpty {
            try await triggerMerge()
        }
    }

    public func save(to url: URL) async throws {
        try checkBackgroundMergeError()
        try await flush()

        guard let base else {
            throw ANNSError.constructionFailed("Nothing to save — index is empty")
        }

        // Snapshot actor state before any cross-actor await points.
        let configSnapshot = config
        let vectorDimensionSnapshot = vectorDimension
        let allVectorDataSnapshot = allVectorData
        let allIDsListSnapshot = allIDsList
        let deletedIDsSnapshot = deletedIDs
        let metadataSnapshot = metadataByID

        let meta = PersistedMeta(
            config: configSnapshot,
            vectorDimension: vectorDimensionSnapshot,
            allVectorData: allVectorDataSnapshot,
            allIDsList: allIDsListSnapshot,
            deletedIDs: Array(deletedIDsSnapshot),
            metadataByID: metadataSnapshot
        )
        try Self.validateLoadedMeta(meta)

        let fileManager = FileManager.default
        let parentURL = url.deletingLastPathComponent()
        let tempURL = parentURL.appendingPathComponent(".\(url.lastPathComponent).tmp-\(UUID().uuidString)")
        let tempBaseURL = tempURL.appendingPathComponent("base.anns")

        try fileManager.createDirectory(at: tempURL, withIntermediateDirectories: true)
        do {
            try await base.save(to: tempBaseURL)

            let dbPath = tempURL.appendingPathComponent("streaming.db").path
            let db = try StreamingDatabase(path: dbPath)

            var slicedVectors: [[Float]] = []
            if let dim = vectorDimensionSnapshot, dim > 0 {
                slicedVectors.reserveCapacity(allIDsListSnapshot.count)
                for i in 0..<allIDsListSnapshot.count {
                    let start = i * dim
                    let end = start + dim
                    slicedVectors.append(Array(allVectorDataSnapshot[start..<end]))
                }
            }
            try db.insertVectors(slicedVectors, ids: allIDsListSnapshot)
            try db.saveConfig(configSnapshot)
            try db.markDeleted(ids: deletedIDsSnapshot)

            if let dim = vectorDimensionSnapshot {
                try db.saveVectorDimension(dim)
            }

            let encoder = JSONEncoder()
            var stringMetadata: [String: [String: String]] = [:]
            for (id, entries) in metadataSnapshot {
                var converted: [String: String] = [:]
                converted.reserveCapacity(entries.count)
                for (key, value) in entries {
                    let data = try encoder.encode(value)
                    guard let json = String(data: data, encoding: .utf8) else {
                        throw ANNSError.constructionFailed("Failed to encode metadata value")
                    }
                    converted[key] = json
                }
                stringMetadata[id] = converted
            }
            try db.saveAllVectorMetadata(stringMetadata)

            try Self.replaceDirectory(at: url, with: tempURL)
        } catch {
            try? fileManager.removeItem(at: tempURL)
            throw error
        }
    }

    public static func load(from url: URL) async throws -> _StreamingIndex {
        let dbURL = url.appendingPathComponent("streaming.db")
        let baseANNSPath = url.appendingPathComponent("base.anns").path
        let hasFreshDB = hasFreshStreamingDatabase(at: dbURL.path, forBaseANNS: baseANNSPath)

        let meta: PersistedMeta
        if hasFreshDB {
            do {
                let db = try StreamingDatabase(path: dbURL.path)
                guard let config = try db.loadConfig() else {
                    throw ANNSError.corruptFile("Missing streaming config in database")
                }

                let (vectors, ids) = try db.loadAllVectors()
                let deletedIDs = try db.loadDeletedIDs()
                let allStringMetadata = try db.loadAllVectorMetadata()
                let dimension = try db.loadVectorDimension()

                let decoder = JSONDecoder()
                var metadataByID: [String: [String: StreamingMetadataValue]] = [:]
                for (id, entries) in allStringMetadata {
                    var converted: [String: StreamingMetadataValue] = [:]
                    converted.reserveCapacity(entries.count)
                    for (key, json) in entries {
                        converted[key] = try decoder.decode(StreamingMetadataValue.self, from: Data(json.utf8))
                    }
                    metadataByID[id] = converted
                }

                let flatVectorData = vectors.flatMap { $0 }
                meta = PersistedMeta(
                    config: config,
                    vectorDimension: dimension,
                    allVectorData: flatVectorData,
                    allIDsList: ids,
                    deletedIDs: Array(deletedIDs),
                    metadataByID: metadataByID
                )
            } catch {
                meta = try loadLegacyMeta(from: url)
            }
        } else {
            meta = try loadLegacyMeta(from: url)
        }
        try validateLoadedMeta(meta)

        let loadedBase = try await GraphIndex.load(from: url.appendingPathComponent("base.anns"))
        let streaming = _StreamingIndex(config: meta.config)
        try await streaming.applyLoadedState(base: loadedBase, meta: meta)
        return streaming
    }

    private static func loadLegacyMeta(from url: URL) throws -> PersistedMeta {
        let sqliteMetaURL = url.appendingPathComponent("streaming.meta.db")
        do {
            if let sqliteMeta = try SQLiteStructuredStore.load(PersistedMeta.self, from: sqliteMetaURL) {
                return sqliteMeta
            }
        } catch {
            // Continue to JSON fallback for backwards compatibility.
        }

        let jsonURL = url.appendingPathComponent("streaming.meta.json")
        guard FileManager.default.fileExists(atPath: jsonURL.path) else {
            throw ANNSError.corruptFile("Missing both streaming.db and streaming.meta.json")
        }
        let data = try Data(contentsOf: jsonURL)
        return try JSONDecoder().decode(PersistedMeta.self, from: data)
    }

    private func applyLoadedState(base: GraphIndex, meta: PersistedMeta) async throws {
        let baseActiveIDs = Set(try await base.streamingActiveExternalIDs())
        let deleted = Set(meta.deletedIDs)
        let pending = Self.pendingRecordsFromMeta(
            meta,
            excludingBaseIDs: baseActiveIDs,
            deletedIDs: deleted
        )
        let pendingIDSet = Set(pending.ids)
        let knownIDs = baseActiveIDs.union(pendingIDSet).union(deleted)

        try Self.validateLoadedState(meta, knownIDs: knownIDs)

        self.base = base
        self.delta = nil
        self.mergeTask = nil
        self._isMerging = false
        self.pendingVectors = []
        self.pendingIDs = []

        self.allVectorData = pending.vectors.flatMap { $0 }
        self.allIDsList = pending.ids
        self.allIDs = knownIDs
        self.deletedIDs = deleted
        self.metadataByID = meta.metadataByID
        self.vectorDimension = meta.vectorDimension

        self.idInBase = baseActiveIDs
        self.idInDelta = []

        if pending.ids.count >= 2 {
            let loadedDelta = try await buildIndex(vectors: pending.vectors, ids: pending.ids)
            self.delta = loadedDelta
            self.idInDelta = Set(pending.ids)
        } else if pending.ids.count == 1 {
            self.pendingVectors = pending.vectors
            self.pendingIDs = pending.ids
        }

        self.metricsNeedsPropagation = true
    }

    private func validateDimension(of vector: [Float]) throws {
        if vector.isEmpty {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        if let expected = vectorDimension {
            guard vector.count == expected else {
                throw ANNSError.dimensionMismatch(expected: expected, got: vector.count)
            }
        } else {
            vectorDimension = vector.count
        }
    }

    private func validateQueryDimension(_ query: [Float]) throws {
        guard let expected = vectorDimension else {
            throw ANNSError.indexEmpty
        }
        guard query.count == expected else {
            throw ANNSError.dimensionMismatch(expected: expected, got: query.count)
        }
    }

    private func removePendingID(_ id: String) {
        if let index = pendingIDs.firstIndex(of: id) {
            pendingIDs.remove(at: index)
            pendingVectors.remove(at: index)
        }
    }

    private func removeFromPendingHistory(_ id: String) {
        guard let historyIndex = allIDsList.firstIndex(of: id) else {
            return
        }

        allIDsList.remove(at: historyIndex)
        guard let dim = vectorDimension else {
            return
        }

        let start = historyIndex * dim
        let end = start + dim
        guard start >= 0, end <= allVectorData.count else {
            allVectorData.removeAll(keepingCapacity: true)
            return
        }
        allVectorData.removeSubrange(start..<end)
    }

    private func adjustedConfiguration(for nodeCount: Int) -> IndexConfiguration {
        var adjusted = config.indexConfiguration
        adjusted.degree = min(adjusted.degree, max(1, nodeCount - 1))
        return adjusted
    }

    private func flushPendingIntoDelta() async throws {
        while true {
            switch StreamingMergePlanner.planDeltaOverflow(captureLedger()) {
            case .idle:
                return

            case .rebuildDelta(let records):
                let newDelta = try await buildIndex(vectors: records.vectors, ids: records.ids)
                delta = newDelta
                idInDelta = Set(records.ids)
                pendingVectors.removeAll(keepingCapacity: true)
                pendingIDs.removeAll(keepingCapacity: true)

            case .appendToDelta(let records):
                guard let delta else {
                    return
                }
                do {
                    try await delta.batchInsert(records.vectors, ids: records.ids)
                    idInDelta.formUnion(records.ids)
                    pendingVectors.removeAll(keepingCapacity: true)
                    pendingIDs.removeAll(keepingCapacity: true)
                } catch let error as ANNSError {
                    switch StreamingMergePlanner.planOverflowRecovery(for: error, isMerging: _isMerging) {
                    case .rethrow:
                        throw error
                    case .waitForCurrentMerge:
                        return
                    case .triggerMerge:
                        try await triggerMerge()
                    }
                }
            }
        }
    }

    private func captureLedger() -> StreamingLedger {
        StreamingLedger(
            pendingVectors: pendingVectors,
            pendingIDs: pendingIDs,
            historyVectorData: allVectorData,
            historyIDs: allIDsList,
            allIDs: allIDs,
            deletedIDs: deletedIDs,
            idInBase: idInBase,
            idInDelta: idInDelta,
            metadataByID: metadataByID,
            vectorDimension: vectorDimension,
            hasDelta: delta != nil
        )
    }

    private func shouldMerge() async -> Bool {
        if _isMerging {
            return false
        }
        guard let delta else {
            return false
        }
        return await delta.count >= config.deltaCapacity
    }

    private func maybeTriggerMerge() async throws {
        guard await shouldMerge() else {
            return
        }

        switch config.mergeStrategy {
        case .blocking:
            try await triggerMerge()
        case .background:
            startBackgroundMergeIfNeeded()
        }
    }

    private func startBackgroundMergeIfNeeded() {
        guard mergeTask == nil else {
            return
        }

        let task = Task { [self] in
            try await self.triggerMerge()
        }
        mergeTask = task

        Task { [self] in
            do {
                try await task.value
            } catch let error as ANNSError {
                self.recordBackgroundMergeError(error)
            } catch {
                self.recordBackgroundMergeError(
                    .constructionFailed("Background merge failed: \(error)")
                )
            }
            self.clearMergeTaskReference()
        }
    }

    private func clearMergeTaskReference() {
        mergeTask = nil
    }

    private func recordBackgroundMergeError(_ error: ANNSError) {
        lastBackgroundMergeError = error
    }

    private func buildIndex(vectors: [[Float]], ids: [String]) async throws -> GraphIndex {
        let index = GraphIndex(configuration: adjustedConfiguration(for: vectors.count))
        try await index.build(vectors: vectors, ids: ids)
        if let metrics {
            await index.setMetrics(metrics)
        }
        try await applyStoredMetadata(to: index, ids: ids)
        return index
    }

    private func applyStoredMetadata(to index: GraphIndex, ids: [String]) async throws {
        for id in ids {
            guard let row = metadataByID[id] else {
                continue
            }

            for (column, value) in row {
                switch value {
                case .string(let value):
                    try await index.setMetadata(column, value: value, for: id)
                case .float(let value):
                    try await index.setMetadata(column, value: value, for: id)
                case .int64(let value):
                    try await index.setMetadata(column, value: value, for: id)
                }
            }
        }
    }

    private func triggerMerge() async throws {
        guard !_isMerging else {
            return
        }
        _isMerging = true
        defer { _isMerging = false }

        let snapshotCount = allIDsList.count
        let prefix = captureLedger().activeRecords(upperBound: snapshotCount)

        let baseRecords: StreamingRecordSet
        if let base {
            let baseSnapshot = try await base.streamingActiveRecords()
            baseRecords = StreamingRecordSet(vectors: baseSnapshot.vectors, ids: baseSnapshot.ids)
        } else {
            baseRecords = .empty
        }

        let plan = StreamingMergePlanner.planMerge(
            baseSnapshotCount: snapshotCount,
            baseRecords: baseRecords,
            prefix: prefix,
            tailLedger: captureLedger()
        )

        switch plan {
        case .resetToEmpty:
            applyResetToEmptyPlan()

        case .rebuildDelta(let retained):
            try await applyRebuildDeltaPlan(retained)

        case .replaceBase(let merged):
            try await applyReplaceBasePlan(merged, tailStartIndex: snapshotCount)
        }
    }

    private func applyResetToEmptyPlan() {
        base = nil
        delta = nil
        idInBase.removeAll()
        idInDelta.removeAll()
        pendingVectors.removeAll(keepingCapacity: true)
        pendingIDs.removeAll(keepingCapacity: true)
        allIDsList.removeAll(keepingCapacity: true)
        allVectorData.removeAll(keepingCapacity: true)
        lastBackgroundMergeError = nil
    }

    private func applyRebuildDeltaPlan(_ retained: StreamingRecordSet) async throws {
        base = nil
        delta = nil
        idInBase.removeAll()
        idInDelta.removeAll()

        allIDsList = retained.ids
        allVectorData = retained.vectors.flatMap { $0 }

        pendingVectors.removeAll(keepingCapacity: true)
        pendingIDs.removeAll(keepingCapacity: true)
        try await applyTailDisposition(StreamingMergePlanner.planTailDisposition(retained))
        lastBackgroundMergeError = nil
    }

    private func applyReplaceBasePlan(_ merged: StreamingRecordSet, tailStartIndex: Int) async throws {
        let newBase = try await buildIndex(vectors: merged.vectors, ids: merged.ids)

        base = newBase
        idInBase = Set(merged.ids)
        delta = nil
        idInDelta.removeAll()
        pendingVectors.removeAll(keepingCapacity: true)
        pendingIDs.removeAll(keepingCapacity: true)

        let tailLedger = captureLedger()
        let tail = tailLedger.activeRecords(in: tailStartIndex..<tailLedger.historyIDs.count)
        allIDsList = tail.ids
        allVectorData = tail.vectors.flatMap { $0 }

        try await applyTailDisposition(StreamingMergePlanner.planTailDisposition(tail))

        if let metrics {
            await metrics.recordMerge()
        }
        lastBackgroundMergeError = nil
    }

    private func applyTailDisposition(_ disposition: StreamingTailDisposition) async throws {
        switch disposition {
        case .discard:
            break
        case .parkPending(let records):
            pendingVectors = records.vectors
            pendingIDs = records.ids
        case .rebuildDelta(let records):
            let newDelta = try await buildIndex(vectors: records.vectors, ids: records.ids)
            delta = newDelta
            idInDelta = Set(records.ids)
        }
    }

    private func pendingSearchResults(
        query: [Float],
        filter: _LegacySearchFilter?,
        metric: Metric
    ) -> [SearchResult] {
        var results: [SearchResult] = []
        results.reserveCapacity(pendingVectors.count)

        for (vector, id) in zip(pendingVectors, pendingIDs) {
            guard !deletedIDs.contains(id), matchesFilter(for: id, filter: filter) else {
                continue
            }

            let score = SIMDDistance.distance(query, vector, metric: metric)
            results.append(SearchResult(id: id, score: score, internalID: UInt32.max))
        }

        return results
    }

    private func dedupeAndSort(_ results: [SearchResult]) -> [SearchResult] {
        var bestByID: [String: SearchResult] = [:]
        bestByID.reserveCapacity(results.count)

        for result in results {
            if let existing = bestByID[result.id] {
                if result.score < existing.score {
                    bestByID[result.id] = result
                }
            } else {
                bestByID[result.id] = result
            }
        }

        return bestByID.values.sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.id < rhs.id
            }
            return lhs.score < rhs.score
        }
    }

    private func matchesFilter(for id: String, filter: _LegacySearchFilter?) -> Bool {
        guard let filter else {
            return true
        }
        let row = metadataByID[id] ?? [:]
        return filter.evaluate(
            stringValue: { column in
                if case .string(let value)? = row[column] {
                    return value
                }
                return nil
            },
            floatValue: { column in
                if case .float(let value)? = row[column] {
                    return value
                }
                return nil
            },
            intValue: { column in
                if case .int64(let value)? = row[column] {
                    return value
                }
                return nil
            }
        )
    }

    private func synchronizeChildMetricsIfNeeded() async {
        guard metricsNeedsPropagation else {
            return
        }
        if let base {
            await base.setMetrics(metrics)
        }
        if let delta {
            await delta.setMetrics(metrics)
        }
        metricsNeedsPropagation = false
    }

    private func checkBackgroundMergeError() throws {
        if let lastBackgroundMergeError {
            throw lastBackgroundMergeError
        }
    }

    private static func validateLoadedMeta(_ meta: PersistedMeta) throws {
        let pendingIDSet = Set(meta.allIDsList)
        guard pendingIDSet.count == meta.allIDsList.count else {
            throw ANNSError.corruptFile("Streaming metadata contains duplicate pending IDs")
        }
        let deletedIDSet = Set(meta.deletedIDs)
        guard deletedIDSet.count == meta.deletedIDs.count else {
            throw ANNSError.corruptFile("Streaming metadata contains duplicate deleted IDs")
        }

        let resolvedDimension = meta.vectorDimension
        if let resolvedDimension {
            guard resolvedDimension > 0 else {
                throw ANNSError.corruptFile("Streaming metadata has invalid vector dimension")
            }
            let expectedFloatCount = try checkedMultiply(meta.allIDsList.count, resolvedDimension)
            guard meta.allVectorData.count == expectedFloatCount else {
                throw ANNSError.corruptFile("Streaming metadata vector payload is inconsistent with dimension and IDs")
            }
        } else if !meta.allVectorData.isEmpty || !meta.allIDsList.isEmpty {
            throw ANNSError.corruptFile("Streaming metadata is missing vector dimension")
        }
    }

    private static func pendingRecordsFromMeta(
        _ meta: PersistedMeta,
        excludingBaseIDs: Set<String>,
        deletedIDs: Set<String>
    ) -> (vectors: [[Float]], ids: [String]) {
        guard let dim = meta.vectorDimension, dim > 0 else {
            return ([], [])
        }

        var vectors: [[Float]] = []
        var ids: [String] = []
        vectors.reserveCapacity(meta.allIDsList.count)
        ids.reserveCapacity(meta.allIDsList.count)

        for (index, id) in meta.allIDsList.enumerated() {
            guard !excludingBaseIDs.contains(id), !deletedIDs.contains(id) else {
                continue
            }
            let start = index * dim
            let end = start + dim
            guard start >= 0, end <= meta.allVectorData.count else {
                continue
            }
            ids.append(id)
            vectors.append(Array(meta.allVectorData[start..<end]))
        }
        return (vectors, ids)
    }

    private static func validateLoadedState(_ meta: PersistedMeta, knownIDs: Set<String>) throws {
        for id in meta.metadataByID.keys where !knownIDs.contains(id) {
            throw ANNSError.corruptFile("Streaming metadata contains row for unknown ID '\(id)'")
        }
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        if overflow {
            throw ANNSError.corruptFile("Streaming metadata size overflow")
        }
        return result
    }

    private nonisolated static func hasFreshStreamingDatabase(
        at dbPath: String,
        forBaseANNS baseANNSPath: String
    ) -> Bool {
        let fm = FileManager.default
        guard
            fm.fileExists(atPath: dbPath),
            fm.fileExists(atPath: baseANNSPath),
            let dbAttrs = try? fm.attributesOfItem(atPath: dbPath),
            let baseAttrs = try? fm.attributesOfItem(atPath: baseANNSPath),
            let dbDate = dbAttrs[.modificationDate] as? Date,
            let baseDate = baseAttrs[.modificationDate] as? Date
        else {
            return false
        }
        return dbDate >= baseDate
    }

    private static func replaceDirectory(at destinationURL: URL, with sourceURL: URL) throws {
        let fileManager = FileManager.default
        let parentURL = destinationURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentURL, withIntermediateDirectories: true)

        if fileManager.fileExists(atPath: destinationURL.path) {
            let backupURL = parentURL.appendingPathComponent(
                ".\(destinationURL.lastPathComponent).backup-\(UUID().uuidString)")
            do {
                try fileManager.moveItem(at: destinationURL, to: backupURL)
                try fileManager.moveItem(at: sourceURL, to: destinationURL)
                try? fileManager.removeItem(at: backupURL)
            } catch {
                if !fileManager.fileExists(atPath: destinationURL.path),
                    fileManager.fileExists(atPath: backupURL.path)
                {
                    try? fileManager.moveItem(at: backupURL, to: destinationURL)
                }
                throw error
            }
        } else {
            try fileManager.moveItem(at: sourceURL, to: destinationURL)
        }
    }
}
