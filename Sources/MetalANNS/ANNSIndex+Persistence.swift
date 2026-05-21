import Foundation
import MetalANNSCore

extension GraphIndex {
    public func save(to url: URL) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }

        let fileManager = FileManager.default
        let parentURL = url.deletingLastPathComponent()
        let tempDirURL = parentURL.appendingPathComponent(".save-tmp-\(UUID().uuidString)")
        let tempANNS = tempDirURL.appendingPathComponent(url.lastPathComponent)
        let dbURL = URL(fileURLWithPath: Self.databasePath(for: url))
        let tempDB = tempDirURL.appendingPathComponent(dbURL.lastPathComponent)

        do {
            try fileManager.createDirectory(at: tempDirURL, withIntermediateDirectories: true)

            try IndexSerializer.save(
                vectors: vectors,
                graph: graph,
                idMap: idMap,
                entryPoint: entryPoint,
                metric: configuration.metric,
                to: tempANNS
            )

            var db: IndexDatabase? = try IndexDatabase(path: tempDB.path)
            try db?.saveIDMap(idMap)
            try db?.saveConfiguration(configuration)
            try db?.saveSoftDeletion(softDeletion)
            try db?.saveMetadataStore(metadataStore)
            try db?.prepareForFileMove()
            db = nil

            try Self.replaceFile(at: url, with: tempANNS)
            try Self.replaceSQLiteFiles(at: dbURL, with: tempDB)
            Self.removeLegacyMetadataSidecars(for: url)
            try? fileManager.removeItem(at: tempDirURL)
        } catch {
            try? fileManager.removeItem(at: tempDirURL)
            throw error
        }
    }

    public func saveMmapCompatible(to url: URL) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }

        let fileManager = FileManager.default
        let parentURL = url.deletingLastPathComponent()
        let tempDirURL = parentURL.appendingPathComponent(".save-tmp-\(UUID().uuidString)")
        let tempANNS = tempDirURL.appendingPathComponent(url.lastPathComponent)
        let dbURL = URL(fileURLWithPath: Self.databasePath(for: url))
        let tempDB = tempDirURL.appendingPathComponent(dbURL.lastPathComponent)

        do {
            try fileManager.createDirectory(at: tempDirURL, withIntermediateDirectories: true)

            try IndexSerializer.saveMmapCompatible(
                vectors: vectors,
                graph: graph,
                idMap: idMap,
                entryPoint: entryPoint,
                metric: configuration.metric,
                to: tempANNS
            )

            var db: IndexDatabase? = try IndexDatabase(path: tempDB.path)
            try db?.saveIDMap(idMap)
            try db?.saveConfiguration(configuration)
            try db?.saveSoftDeletion(softDeletion)
            try db?.saveMetadataStore(metadataStore)
            try db?.prepareForFileMove()
            db = nil

            try Self.replaceFile(at: url, with: tempANNS)
            try Self.replaceSQLiteFiles(at: dbURL, with: tempDB)
            Self.removeLegacyMetadataSidecars(for: url)
            try? fileManager.removeItem(at: tempDirURL)
        } catch {
            try? fileManager.removeItem(at: tempDirURL)
            throw error
        }
    }

    public static func load(from url: URL) async throws -> GraphIndex {
        let persistedState = try resolvePersistedState(for: url)
        let initialConfiguration = persistedState.configuration ?? .default
        let index = GraphIndex(configuration: initialConfiguration)

        let loaded = try IndexSerializer.load(from: url, device: await index.currentDevice())
        let resolvedIDMap = resolveLoadedIDMap(
            persistedIDMap: persistedState.idMap,
            serializerIDMap: loaded.idMap
        )

        var resolvedConfiguration = persistedState.configuration ?? .default
        resolvedConfiguration.metric = loaded.metric
        resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16
        resolvedConfiguration.useBinary = loaded.vectors is BinaryVectorBuffer

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: resolvedIDMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedState.softDeletion,
            metadataStore: persistedState.metadataStore
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public static func loadMmap(from url: URL) async throws -> GraphIndex {
        let persistedState = try resolvePersistedState(for: url)
        let initialConfiguration = persistedState.configuration ?? .default
        let index = GraphIndex(configuration: initialConfiguration)
        let loaded = try MmapIndexLoader.load(from: url, device: await index.currentDevice())
        let resolvedIDMap = resolveLoadedIDMap(
            persistedIDMap: persistedState.idMap,
            serializerIDMap: loaded.idMap
        )

        var resolvedConfiguration = persistedState.configuration ?? .default
        resolvedConfiguration.metric = loaded.metric
        resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16
        resolvedConfiguration.useBinary = loaded.isBinary

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: resolvedIDMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedState.softDeletion,
            metadataStore: persistedState.metadataStore,
            isReadOnlyLoadedIndex: true,
            mmapLifetime: loaded.mmapLifetime
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public static func loadDiskBacked(from url: URL) async throws -> GraphIndex {
        let persistedState = try resolvePersistedState(for: url)
        let initialConfiguration = persistedState.configuration ?? .default
        let index = GraphIndex(configuration: initialConfiguration)

        let diskBacked = try DiskBackedIndexLoader.load(from: url, device: await index.currentDevice())
        let resolvedIDMap = resolveLoadedIDMap(
            persistedIDMap: persistedState.idMap,
            serializerIDMap: diskBacked.idMap
        )

        var resolvedConfiguration = persistedState.configuration ?? .default
        resolvedConfiguration.metric = diskBacked.metric
        resolvedConfiguration.useFloat16 = diskBacked.vectors.isFloat16
        resolvedConfiguration.useBinary = diskBacked.isBinary

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: diskBacked.vectors,
            graph: diskBacked.graph,
            idMap: resolvedIDMap,
            entryPoint: diskBacked.entryPoint,
            softDeletion: persistedState.softDeletion,
            metadataStore: persistedState.metadataStore,
            isReadOnlyLoadedIndex: true,
            mmapLifetime: diskBacked.mmapLifetime
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public var count: Int {
        max(0, idMap.count - softDeletion.deletedCount)
    }

    func streamingActiveExternalIDs() throws -> [String] {
        guard isBuilt, let vectors else {
            throw ANNSError.indexEmpty
        }

        var activeIDs: [String] = []
        activeIDs.reserveCapacity(max(0, idMap.count - softDeletion.deletedCount))
        for slot in 0..<vectors.count {
            let internalID = UInt32(slot)
            guard !softDeletion.isDeleted(internalID) else {
                continue
            }
            guard let externalID = idMap.externalID(for: internalID) else {
                continue
            }
            activeIDs.append(externalID)
        }
        return activeIDs
    }

    func streamingActiveRecords() throws -> (vectors: [[Float]], ids: [String]) {
        guard isBuilt, let vectors else {
            throw ANNSError.indexEmpty
        }

        var activeVectors: [[Float]] = []
        var activeIDs: [String] = []
        activeVectors.reserveCapacity(max(0, idMap.count - softDeletion.deletedCount))
        activeIDs.reserveCapacity(max(0, idMap.count - softDeletion.deletedCount))

        for slot in 0..<vectors.count {
            let internalID = UInt32(slot)
            guard !softDeletion.isDeleted(internalID) else {
                continue
            }
            guard let externalID = idMap.externalID(for: internalID) else {
                continue
            }
            activeIDs.append(externalID)
            activeVectors.append(vectors.vector(at: slot))
        }

        return (activeVectors, activeIDs)
    }

}
