import Foundation
import MetalANNSCore

extension GraphIndex {
    public func insert(_ vector: [Float], id: String) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard vector.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
        }
        if idMap.internalID(for: id) != nil {
            throw ANNSError.idAlreadyExists(id)
        }

        guard idMap.canAllocate(1) else {
            throw ANNSError.constructionFailed("Internal ID space exhausted")
        }

        let nextInternalID = Int(idMap.nextInternalID)
        guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector
        let slot = nextInternalID
        let previousVectorCount = vectors.count
        let previousGraphCount = graph.nodeCount

        let metricsRecorder = metrics
        let insertStart = metricsRecorder == nil ? nil : ContinuousClock.now

        do {
            try vectors.insert(vector: graphVector, at: slot)
            if vectors.count < slot + 1 {
                vectors.setCount(slot + 1)
            }

            try IncrementalBuilder.insert(
                vector: graphVector,
                at: slot,
                into: graph,
                vectors: vectors,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )
            if graph.nodeCount < slot + 1 {
                graph.setCount(slot + 1)
            }
            hnsw = nil

            let repairConfig = configuration.repairConfiguration
            if repairConfig.enabled && repairConfig.repairInterval > 0 {
                pendingRepairIDs.append(UInt32(slot))
                if pendingRepairIDs.count >= repairConfig.repairInterval {
                    try triggerRepair(throwOnFailure: false)
                }
            }

            guard let assignedID = idMap.assign(externalID: id), Int(assignedID) == slot else {
                throw ANNSError.constructionFailed("Failed to commit internal ID for '\(id)'")
            }
        } catch {
            vectors.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)

            // Best-effort cleanup of the uncommitted node slot.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Incremental insert failed: \(error)")
        }

        if let metricsRecorder, let insertStart {
            let duration = ContinuousClock.now - insertStart
            await metricsRecorder.recordInsert(durationNs: Self.durationNanoseconds(duration))
        }
    }

    /// Inserts a vector with a numeric (UInt64) key.
    /// For use by Wax's UInt64 frameId-based API.
    public func insert(_ vector: [Float], numericID: UInt64) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard vector.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
        }
        if idMap.internalID(forNumeric: numericID) != nil {
            throw ANNSError.idAlreadyExists(String(numericID))
        }

        guard idMap.canAllocate(1) else {
            throw ANNSError.constructionFailed("Internal ID space exhausted")
        }

        let nextInternalID = Int(idMap.nextInternalID)
        guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector
        let slot = nextInternalID
        let previousVectorCount = vectors.count
        let previousGraphCount = graph.nodeCount

        let metricsRecorder = metrics
        let insertStart = metricsRecorder == nil ? nil : ContinuousClock.now

        do {
            try vectors.insert(vector: graphVector, at: slot)
            if vectors.count < slot + 1 {
                vectors.setCount(slot + 1)
            }

            try IncrementalBuilder.insert(
                vector: graphVector,
                at: slot,
                into: graph,
                vectors: vectors,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )
            if graph.nodeCount < slot + 1 {
                graph.setCount(slot + 1)
            }
            hnsw = nil

            let repairConfig = configuration.repairConfiguration
            if repairConfig.enabled && repairConfig.repairInterval > 0 {
                pendingRepairIDs.append(UInt32(slot))
                if pendingRepairIDs.count >= repairConfig.repairInterval {
                    try triggerRepair(throwOnFailure: false)
                }
            }

            guard let assignedID = idMap.assign(numericID: numericID), Int(assignedID) == slot else {
                throw ANNSError.constructionFailed("Failed to commit internal ID for numeric \(numericID)")
            }
        } catch {
            vectors.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)

            // Best-effort cleanup of the uncommitted node slot.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Incremental insert (numeric) failed: \(error)")
        }

        if let metricsRecorder, let insertStart {
            let duration = ContinuousClock.now - insertStart
            await metricsRecorder.recordInsert(durationNs: Self.durationNanoseconds(duration))
        }
    }

    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
        guard isBuilt, let vectorStorage = self.vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }

        let dim = vectorStorage.dim
        for vector in vectors {
            guard vector.count == dim else {
                throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
            }
        }

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
            if idMap.internalID(for: id) != nil {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        guard idMap.canAllocate(ids.count) else {
            throw ANNSError.constructionFailed("Internal ID space exhausted")
        }

        let startSlot = Int(idMap.nextInternalID)
        guard startSlot + vectors.count <= vectorStorage.capacity,
            startSlot + vectors.count <= graph.capacity
        else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        let slots = Array(startSlot..<(startSlot + vectors.count))
        let previousVectorCount = vectorStorage.count
        let previousGraphCount = graph.nodeCount

        let metricsRecorder = metrics
        let batchInsertStart = metricsRecorder == nil ? nil : ContinuousClock.now
        let insertedCount = vectors.count

        do {
            let graphVectors: [[Float]]
            if vectorStorage is BinaryVectorBuffer {
                graphVectors = vectors.map(Self.quantizeForHamming)
            } else {
                graphVectors = vectors
            }

            for (offset, vector) in graphVectors.enumerated() {
                try vectorStorage.insert(vector: vector, at: slots[offset])
            }
            let newMaxCount = (slots.last ?? 0) + 1
            if vectorStorage.count < newMaxCount {
                vectorStorage.setCount(newMaxCount)
            }

            try BatchIncrementalBuilder.batchInsert(
                vectors: graphVectors,
                startingAt: startSlot,
                into: graph,
                vectorStorage: vectorStorage,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )

            if graph.nodeCount < newMaxCount {
                graph.setCount(newMaxCount)
            }
            hnsw = nil

            let repairConfig = configuration.repairConfiguration
            if repairConfig.enabled {
                pendingRepairIDs.append(contentsOf: slots.map(UInt32.init))
                try triggerRepair(throwOnFailure: false)
            }

            for (offset, id) in ids.enumerated() {
                guard let assignedID = idMap.assign(externalID: id), Int(assignedID) == slots[offset] else {
                    throw ANNSError.constructionFailed("Failed to commit internal ID for '\(id)'")
                }
            }
        } catch {
            vectorStorage.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)
            pendingRepairIDs.removeAll { internalID in
                slots.contains(Int(internalID))
            }

            // Best-effort cleanup for uncommitted slots.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            for slot in slots {
                try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)
            }

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Batch insert failed: \(error)")
        }

        if let metricsRecorder, let batchInsertStart {
            let duration = ContinuousClock.now - batchInsertStart
            await metricsRecorder.recordBatchInsert(
                count: insertedCount,
                durationNs: Self.durationNanoseconds(duration)
            )
        }
    }

    public func repair() throws(ANNSError) {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard isBuilt, vectors != nil, graph != nil else {
            throw ANNSError.indexEmpty
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }
        try triggerRepair()
    }

    private func triggerRepair(throwOnFailure: Bool = true) throws(ANNSError) {
        guard let vectors, let graph else {
            return
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }

        let idsToRepair = pendingRepairIDs
        pendingRepairIDs.removeAll(keepingCapacity: true)

        do {
            _ = try GraphRepairer.repair(
                recentIDs: idsToRepair,
                vectors: vectors,
                graph: graph,
                config: configuration.repairConfiguration,
                metric: configuration.metric
            )
            hnsw = nil
        } catch {
            // Preserve pending IDs so repair can be retried later.
            pendingRepairIDs = idsToRepair + pendingRepairIDs
            if throwOnFailure {
                throw error
            }
        }
    }

    public func delete(id: String) throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        softDeletion.markDeleted(internalID)
        metadataStore.remove(id: internalID)
    }

    public func compact() async throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard softDeletion.deletedCount > 0 else {
            return
        }

        let result = try await IndexCompactor.compact(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            softDeletion: softDeletion,
            metric: configuration.metric,
            degree: configuration.degree,
            context: context,
            maxIterations: configuration.maxIterations,
            convergenceThreshold: configuration.convergenceThreshold,
            useFloat16: configuration.useFloat16,
            useBinary: configuration.useBinary
        )

        var remapping: [UInt32: UInt32] = [:]
        remapping.reserveCapacity(result.idMap.count)
        for oldIndex in 0..<vectors.count {
            let oldID = UInt32(oldIndex)
            if softDeletion.isDeleted(oldID) {
                continue
            }
            let newID: UInt32?
            if let externalID = idMap.externalID(for: oldID) {
                newID = result.idMap.internalID(for: externalID)
            } else if let numericID = idMap.numericID(for: oldID) {
                newID = result.idMap.internalID(forNumeric: numericID)
            } else {
                newID = nil
            }
            guard let newID else {
                continue
            }
            remapping[oldID] = newID
        }

        self.vectors = result.vectors
        self.graph = result.graph
        self.idMap = result.idMap
        self.entryPoint = result.entryPoint
        self.softDeletion = SoftDeletion()
        self.metadataStore = metadataStore.remapped(using: remapping)
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
    }

    public func setMetadata(_ column: String, value: String, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, stringValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Float, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, floatValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Int64, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, intValue: value, for: internalID)
    }

}
