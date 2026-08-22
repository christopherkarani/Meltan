import Foundation
import Metal
import MetalANNSCore

extension GraphIndex {
    internal func batchSearchMaxConcurrency() async -> Int {
        if let context {
            return await context.queuePool.queues.count
        }
        return max(1, ProcessInfo.processInfo.activeProcessorCount)
    }

    internal func currentDevice() -> MTLDevice? {
        context?.device
    }

    internal func shouldUseGPUConstruction(nodeCount: Int) -> Bool {
        guard context != nil else {
            return false
        }
        guard !configuration.useBinary, configuration.metric != .hamming else {
            return false
        }
        return nodeCount >= Self.minGPUConstructionNodeCount
    }

    /// Lazily builds the HNSW overlay before a CPU-path search, memoizing it
    /// on the actor. Called through `VectorSearch.execute`'s `ensureHNSW` hook.
    internal func materializeHNSWForFallback() throws -> HNSWLayers? {
        if configuration.hnswConfiguration.enabled, hnsw == nil {
            try rebuildHNSWFromCurrentState()
        }
        return hnsw
    }

    internal func applyLoadedState(
        configuration: IndexConfiguration,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        softDeletion: SoftDeletion,
        metadataStore: MetadataStore = MetadataStore(),
        isReadOnlyLoadedIndex: Bool = false,
        mmapLifetime: AnyObject? = nil
    ) {
        self.configuration = configuration
        self.vectors = vectors
        self.graph = graph
        self.idMap = idMap
        self.entryPoint = entryPoint
        self.softDeletion = softDeletion
        self.metadataStore = metadataStore
        self.isBuilt = true
        self.isReadOnlyLoadedIndex = isReadOnlyLoadedIndex
        self.mmapLifetime = mmapLifetime
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
    }

    internal func rebuildHNSWFromCurrentState() throws(ANNSError) {
        guard configuration.hnswConfiguration.enabled else {
            hnsw = nil
            return
        }
        guard let vectors, let graph else {
            hnsw = nil
            return
        }
        guard vectors.count > 0, graph.nodeCount > 0 else {
            hnsw = nil
            return
        }

        hnsw = try HNSWBuilder.buildLayers(
            vectors: vectors,
            graph: VectorSearch.extractGraph(from: graph),
            nodeCount: vectors.count,
            metric: configuration.metric,
            config: configuration.hnswConfiguration
        )
    }

    internal nonisolated static func quantizeForHamming(_ vector: [Float]) -> [Float] {
        vector.map { $0 >= 0 ? 1.0 : 0.0 }
    }

    internal nonisolated static func metadataURL(for fileURL: URL) -> URL {
        URL(fileURLWithPath: fileURL.path + ".meta.json")
    }

    internal nonisolated static func metadataDBURL(for fileURL: URL) -> URL {
        URL(fileURLWithPath: fileURL.path + ".meta.db")
    }

    internal nonisolated static func removeLegacyMetadataSidecars(for fileURL: URL) {
        let fileManager = FileManager.default
        let metadataDB = metadataDBURL(for: fileURL)
        let sidecars = [
            metadataURL(for: fileURL),
            metadataDB,
            URL(fileURLWithPath: metadataDB.path + "-wal"),
            URL(fileURLWithPath: metadataDB.path + "-shm"),
        ]
        for sidecar in sidecars where fileManager.fileExists(atPath: sidecar.path) {
            try? fileManager.removeItem(at: sidecar)
        }
    }

    internal nonisolated static func databasePath(for fileURL: URL) -> String {
        fileURL.deletingPathExtension().appendingPathExtension("db").path
    }

    internal nonisolated static func replaceFile(at destination: URL, with source: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: source)
        } else {
            try fm.moveItem(at: source, to: destination)
        }
    }

    internal nonisolated static func replaceSQLiteFiles(at destinationDB: URL, with sourceDB: URL) throws {
        let fm = FileManager.default

        func replace(_ source: URL, _ destination: URL) throws {
            guard fm.fileExists(atPath: source.path) else {
                return
            }
            if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(destination, withItemAt: source)
            } else {
                try fm.moveItem(at: source, to: destination)
            }
        }

        func replaceSidecar(suffix: String) throws {
            let source = URL(fileURLWithPath: sourceDB.path + suffix)
            let destination = URL(fileURLWithPath: destinationDB.path + suffix)
            if fm.fileExists(atPath: source.path) {
                try replace(source, destination)
            } else if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
        }

        try replace(sourceDB, destinationDB)
        try replaceSidecar(suffix: "-wal")
        try replaceSidecar(suffix: "-shm")
    }

    internal nonisolated static func durationNanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        let seconds = components.seconds > 0 ? UInt64(components.seconds) : 0
        let attoseconds = components.attoseconds > 0 ? UInt64(components.attoseconds) : 0
        return seconds &* 1_000_000_000 &+ attoseconds / 1_000_000_000
    }

    internal nonisolated static func hasFreshDatabase(at dbPath: String, forANNS annsPath: String) -> Bool {
        let fm = FileManager.default
        guard
            fm.fileExists(atPath: dbPath),
            fm.fileExists(atPath: annsPath),
            let dbAttrs = try? fm.attributesOfItem(atPath: dbPath),
            let annsAttrs = try? fm.attributesOfItem(atPath: annsPath),
            let dbDate = dbAttrs[.modificationDate] as? Date,
            let annsDate = annsAttrs[.modificationDate] as? Date
        else {
            return false
        }
        return dbDate >= annsDate
    }

    internal nonisolated static func resolvePersistedState(for fileURL: URL) throws -> (
        configuration: IndexConfiguration?,
        softDeletion: SoftDeletion,
        metadataStore: MetadataStore,
        idMap: IDMap?
    ) {
        let dbPath = databasePath(for: fileURL)
        if hasFreshDatabase(at: dbPath, forANNS: fileURL.path) {
            do {
                let db = try IndexDatabase(path: dbPath)
                return (
                    configuration: try db.loadConfiguration(),
                    softDeletion: try db.loadSoftDeletion(),
                    metadataStore: try db.loadMetadataStore(),
                    idMap: try db.loadIDMap()
                )
            } catch {
                let legacy = try loadPersistedMetadataIfPresent(from: fileURL)
                return (
                    configuration: legacy?.configuration,
                    softDeletion: legacy?.softDeletion ?? SoftDeletion(),
                    metadataStore: legacy?.metadataStore ?? MetadataStore(),
                    idMap: legacy?.idMap
                )
            }
        }

        let legacy = try loadPersistedMetadataIfPresent(from: fileURL)
        return (
            configuration: legacy?.configuration,
            softDeletion: legacy?.softDeletion ?? SoftDeletion(),
            metadataStore: legacy?.metadataStore ?? MetadataStore(),
            idMap: legacy?.idMap
        )
    }

    internal nonisolated static func resolveLoadedIDMap(
        persistedIDMap: IDMap?,
        serializerIDMap: IDMap
    ) -> IDMap {
        if let persistedIDMap, persistedIDMap.count == serializerIDMap.count {
            return persistedIDMap
        }
        return serializerIDMap
    }

    // MARK: - Legacy JSON Sidecar (backward compatibility only)

    /// Loads metadata from legacy sidecar formats.
    /// Used only when no fresh `.db` file exists (pre-migration indexes).
    /// Tries `.meta.db` (SQLiteStructuredStore) first, then `.meta.json`.
    /// Do not call this directly — use `resolvePersistedState(for:)`.
    internal nonisolated static func loadPersistedMetadataIfPresent(from fileURL: URL) throws -> _PersistedMetadata? {
        do {
            if let sqliteMeta = try SQLiteStructuredStore.load(_PersistedMetadata.self, from: metadataDBURL(for: fileURL)) {
                return sqliteMeta
            }
        } catch {
            // Keep JSON fallback path for legacy metadata recovery.
        }

        let metadataURL = metadataURL(for: fileURL)
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            return nil
        }

        let data = try Data(contentsOf: metadataURL)
        return try JSONDecoder().decode(_PersistedMetadata.self, from: data)
    }
}
