import Dispatch
import Foundation
import Metal
import MetalANNSCore

public actor _IVFPQIndex: Sendable {
    private struct PersistedState: Codable, Sendable {
        let capacity: Int
        let dimension: Int
        let config: IVFPQConfiguration
        let coarseCentroids: [[Float]]
        let pq: ProductQuantizer
        let vectorCodes: [[UInt8]]
        let invertedLists: [[UInt32]]
        let coarseAssignments: [UInt32]
        let idMap: IDMap
    }

    private static let persistenceMagic: [UInt8] = [0x49, 0x56, 0x46, 0x50]  // "IVFP"
    private static let persistenceVersion: UInt32 = 1

    private let config: IVFPQConfiguration
    private let capacity: Int
    private let dimension: Int
    private let context: MetalContext?

    private var coarseCentroids: [[Float]] = []
    private var flattenedCoarseCentroids: [Float] = []
    private var pq: ProductQuantizer?
    private var flattenedCodebooks: [Float] = []
    private var vectorBuffer: PQVectorBuffer?
    private var invertedLists: [[UInt32]] = []
    private var coarseAssignments: [UInt32] = []
    private var idMap = IDMap()
    private var isTrained = false

    public init(capacity: Int, dimension: Int, config: IVFPQConfiguration) throws {
        guard capacity > 0 else {
            throw ANNSError.constructionFailed("_IVFPQIndex capacity must be greater than zero")
        }
        guard dimension > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: dimension)
        }
        guard config.numCentroids == 256 else {
            throw ANNSError.constructionFailed("_IVFPQIndex requires numCentroids == 256 (UInt8 codes)")
        }

        self.capacity = capacity
        self.dimension = dimension
        self.config = config
        self.context = try? MetalContext()
    }

    public func train(vectors: [[Float]]) async throws {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot train _IVFPQIndex with empty vectors")
        }
        guard vectors.count >= config.numCentroids else {
            throw ANNSError.constructionFailed("Need at least \(config.numCentroids) vectors for PQ training")
        }
        guard vectors.count >= config.numCoarseCentroids else {
            throw ANNSError.constructionFailed(
                "Need at least \(config.numCoarseCentroids) vectors for coarse quantizer"
            )
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }

        let coarseResult = try KMeans.cluster(
            vectors: vectors,
            k: config.numCoarseCentroids,
            maxIterations: config.trainingIterations,
            metric: config.metric,
            seed: 42
        )
        let trainedCoarseCentroids = coarseResult.centroids

        var residuals: [[Float]] = []
        residuals.reserveCapacity(vectors.count)
        for (vectorIndex, vector) in vectors.enumerated() {
            let cluster = coarseResult.assignments[vectorIndex]
            let centroid = trainedCoarseCentroids[cluster]
            residuals.append(residual(vector: vector, centroid: centroid))
        }

        let trainedPQ = try ProductQuantizer.train(
            vectors: residuals,
            numSubspaces: config.numSubspaces,
            centroidsPerSubspace: config.numCentroids,
            maxIterations: config.trainingIterations
        )

        self.coarseCentroids = trainedCoarseCentroids
        self.flattenedCoarseCentroids = trainedCoarseCentroids.flatMap { $0 }
        self.pq = trainedPQ
        self.flattenedCodebooks = GPUADCSearch.flattenCodebooks(from: trainedPQ)
        self.vectorBuffer = try PQVectorBuffer(capacity: capacity, dim: dimension, pq: trainedPQ)
        self.invertedLists = Array(repeating: [], count: trainedCoarseCentroids.count)
        self.coarseAssignments = []
        self.idMap = IDMap()
        self.isTrained = true
    }

    public func add(vectors: [[Float]], ids: [String]) async throws {
        guard isTrained, let vectorBuffer, let pq else {
            throw ANNSError.constructionFailed("_IVFPQIndex must be trained before add()")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }
        guard idMap.count + vectors.count <= capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded")
        }

        var seen = Set<String>()
        for id in ids {
            if !seen.insert(id).inserted || idMap.internalID(for: id) != nil {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }

        let localCoarseCentroids = coarseCentroids
        let localFlattenedCoarseCentroids = flattenedCoarseCentroids
        let localMetric = config.metric
        let localDimension = dimension
        let planningState = ParallelAddPlanningState(
            vectorCount: vectors.count,
            codeLength: pq.numSubspaces
        )

        DispatchQueue.concurrentPerform(iterations: vectors.count) { offset in
            guard planningState.shouldContinue else {
                return
            }

            do {
                let vector = vectors[offset]
                let cluster = Self.nearestCoarseCentroid(
                    for: vector,
                    flattenedCentroids: localFlattenedCoarseCentroids,
                    centroidCount: localCoarseCentroids.count,
                    dimension: localDimension,
                    metric: localMetric
                )
                let encoded = try pq.encode(vector: vector, subtracting: localCoarseCentroids[cluster])
                planningState.store(cluster: cluster, code: encoded, at: offset)
            } catch {
                planningState.store(error: error)
            }
        }

        if let planningError = planningState.firstError {
            throw planningError
        }

        for offset in vectors.indices {
            guard let internalID = idMap.assign(externalID: ids[offset]) else {
                throw ANNSError.idAlreadyExists(ids[offset])
            }

            let cluster = planningState.clusterAssignments[offset]
            try vectorBuffer.insertEncoded(code: planningState.encodedCodes[offset], at: Int(internalID))
            invertedLists[cluster].append(internalID)
            coarseAssignments.append(UInt32(cluster))
        }
    }

    public func search(query: [Float], k: Int, nprobe: Int? = nil) async -> [SearchResult] {
        await search(query: query, k: k, nprobe: nprobe, forceGPU: nil)
    }

    func search(query: [Float], k: Int, nprobe: Int? = nil, forceGPU: Bool?) async -> [SearchResult] {
        guard isTrained, let vectorBuffer else {
            return []
        }
        guard k > 0 else {
            return []
        }
        guard query.count == dimension else {
            return []
        }
        guard !coarseCentroids.isEmpty else {
            return []
        }

        let probeCount = max(1, min(nprobe ?? config.nprobe, coarseCentroids.count))
        let centroidScores = selectClosestCentroids(to: query, limit: probeCount)
        var merged = BoundedSortedList<SearchResult>(capacity: k) {
            $0.score < $1.score
        }

        guard let pq else {
            return []
        }

        for (clusterIndex, _) in centroidScores {
            let queryResidual = residual(vector: query, centroid: coarseCentroids[clusterIndex])
            let candidateIDs = invertedLists[clusterIndex]
            let scores = await distancesForCluster(
                queryResidual: queryResidual,
                candidateIDs: candidateIDs,
                forceGPU: forceGPU,
                pq: pq,
                vectorBuffer: vectorBuffer
            )

            for (offset, internalID) in candidateIDs.enumerated() {
                let score = scores[offset]
                guard let externalID = idMap.externalID(for: internalID) else {
                    continue
                }
                merged.insert(SearchResult(id: externalID, score: score, internalID: internalID))
            }
        }

        return merged.elements
    }

    public var count: Int {
        idMap.count
    }

    public func estimatedVectorCodeBytes() -> Int {
        vectorBuffer?.compressedCodeBytes ?? 0
    }

    public func estimatedMemoryBytes() -> Int {
        let coarseBytes = coarseCentroids.count * dimension * MemoryLayout<Float>.stride
        let codebookBytes: Int =
            if let pq {
                pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension * MemoryLayout<Float>.stride
            } else {
                0
            }
        let vectorCodeBytes = vectorBuffer?.compressedCodeBytes ?? 0
        let invertedListBytes = invertedLists.reduce(0) { partial, list in
            partial + (list.count * MemoryLayout<UInt32>.stride)
        }
        let assignmentBytes = coarseAssignments.count * MemoryLayout<UInt32>.stride
        return coarseBytes + codebookBytes + vectorCodeBytes + invertedListBytes + assignmentBytes
    }

    public func save(to path: String) async throws {
        guard isTrained, let pq, let vectorBuffer else {
            throw ANNSError.indexEmpty
        }

        let vectorCodes = (0..<count).map { vectorBuffer.code(at: $0) }
        let state = PersistedState(
            capacity: capacity,
            dimension: dimension,
            config: config,
            coarseCentroids: coarseCentroids,
            pq: pq,
            vectorCodes: vectorCodes,
            invertedLists: invertedLists,
            coarseAssignments: coarseAssignments,
            idMap: idMap
        )

        let payload = try JSONEncoder().encode(state)
        var data = Data()
        data.append(contentsOf: Self.persistenceMagic)
        Self.appendUInt32(Self.persistenceVersion, to: &data)
        Self.appendUInt32(UInt32(payload.count), to: &data)
        data.append(payload)

        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
    }

    public static func load(from path: String) async throws -> _IVFPQIndex {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        guard data.count >= 12 else {
            throw ANNSError.corruptFile("IVFPQ persistence payload is too small")
        }

        var cursor = 0
        let magic = Array(data[cursor..<cursor + 4])
        cursor += 4
        guard magic == persistenceMagic else {
            throw ANNSError.corruptFile("Invalid IVFPQ persistence magic")
        }

        let version = try readUInt32(from: data, cursor: &cursor)
        guard version == persistenceVersion else {
            throw ANNSError.corruptFile("Unsupported IVFPQ persistence version \(version)")
        }

        let payloadLength = Int(try readUInt32(from: data, cursor: &cursor))
        guard payloadLength >= 0, cursor + payloadLength <= data.count else {
            throw ANNSError.corruptFile("Invalid IVFPQ persistence payload length")
        }

        let payload = data[cursor..<cursor + payloadLength]
        let state: PersistedState
        do {
            state = try JSONDecoder().decode(PersistedState.self, from: payload)
        } catch {
            throw ANNSError.corruptFile("Invalid IVFPQ persistence JSON payload")
        }

        let index = try _IVFPQIndex(
            capacity: state.capacity,
            dimension: state.dimension,
            config: state.config
        )
        try await index.restore(from: state)
        return index
    }

    func gpuAvailable() -> Bool {
        context != nil
    }

    func firstNonEmptyCluster() -> Int? {
        invertedLists.firstIndex(where: { !$0.isEmpty })
    }

    func debugClusterDistances(query: [Float], clusterIndex: Int, forceGPU: Bool) async throws -> [Float] {
        guard let pq, let vectorBuffer, isTrained else {
            throw ANNSError.searchFailed("Index must be trained before debug distance queries")
        }
        guard query.count == dimension else {
            throw ANNSError.dimensionMismatch(expected: dimension, got: query.count)
        }
        guard clusterIndex >= 0, clusterIndex < coarseCentroids.count else {
            throw ANNSError.searchFailed("Cluster index out of range")
        }

        let residualQuery = residual(vector: query, centroid: coarseCentroids[clusterIndex])
        let candidateIDs = invertedLists[clusterIndex]
        return await distancesForCluster(
            queryResidual: residualQuery,
            candidateIDs: candidateIDs,
            forceGPU: forceGPU,
            pq: pq,
            vectorBuffer: vectorBuffer
        )
    }

    private func nearestCoarseCentroid(for vector: [Float]) -> Int {
        Self.nearestCoarseCentroid(
            for: vector,
            flattenedCentroids: flattenedCoarseCentroids,
            centroidCount: coarseCentroids.count,
            dimension: dimension,
            metric: config.metric
        )
    }

    private func residual(vector: [Float], centroid: [Float]) -> [Float] {
        zip(vector, centroid).map { $0 - $1 }
    }

    private func distancesForCluster(
        queryResidual: [Float],
        candidateIDs: [UInt32],
        forceGPU: Bool?,
        pq: ProductQuantizer,
        vectorBuffer: PQVectorBuffer
    ) async -> [Float] {
        guard !candidateIDs.isEmpty else {
            return []
        }

        if forceGPU == false {
            return cpuADCDistances(
                queryResidual: queryResidual,
                candidateIDs: candidateIDs,
                pq: pq,
                vectorBuffer: vectorBuffer
            )
        }

        let shouldTryGPU: Bool
        if forceGPU == true {
            shouldTryGPU = true
        } else {
            shouldTryGPU = context != nil && candidateIDs.count >= 64
        }

        if shouldTryGPU {
            do {
                return try await gpuADCDistances(
                    queryResidual: queryResidual,
                    candidateIDs: candidateIDs,
                    pq: pq,
                    vectorBuffer: vectorBuffer
                )
            } catch {
                // Fall back to CPU path for robustness.
            }
        }

        return cpuADCDistances(
            queryResidual: queryResidual,
            candidateIDs: candidateIDs,
            pq: pq,
            vectorBuffer: vectorBuffer
        )
    }

    private func cpuADCDistances(
        queryResidual: [Float],
        candidateIDs: [UInt32],
        pq: ProductQuantizer,
        vectorBuffer: PQVectorBuffer
    ) -> [Float] {
        guard let table = pq.distanceTableFlattened(query: queryResidual, metric: config.metric) else {
            return [Float](repeating: Float.greatestFiniteMagnitude, count: candidateIDs.count)
        }

        var distances: [Float] = []
        distances.reserveCapacity(candidateIDs.count)
        for internalID in candidateIDs {
            guard
                let distance = vectorBuffer.withCode(
                    at: Int(internalID),
                    { code in
                        guard code.count == pq.numSubspaces else {
                            return Float.greatestFiniteMagnitude
                        }

                        var total: Float = 0
                        for subspace in 0..<pq.numSubspaces {
                            total += table[(subspace * pq.centroidsPerSubspace) + Int(code[subspace])]
                        }
                        return total
                    })
            else {
                distances.append(Float.greatestFiniteMagnitude)
                continue
            }
            distances.append(distance)
        }
        return distances
    }

    private func gpuADCDistances(
        queryResidual: [Float],
        candidateIDs: [UInt32],
        pq: ProductQuantizer,
        vectorBuffer: PQVectorBuffer
    ) async throws -> [Float] {
        guard let context else {
            throw ANNSError.searchFailed("Metal context unavailable for GPU ADC")
        }
        guard queryResidual.count == dimension else {
            throw ANNSError.dimensionMismatch(expected: dimension, got: queryResidual.count)
        }
        guard !candidateIDs.isEmpty else {
            return []
        }

        if flattenedCodebooks.isEmpty {
            flattenedCodebooks = GPUADCSearch.flattenCodebooks(from: pq)
        }

        return try await GPUADCSearch.computeDistances(
            context: context,
            query: queryResidual,
            pq: pq,
            packedCodes: vectorBuffer.gatherCodes(for: candidateIDs),
            vectorCount: candidateIDs.count,
            flatCodebooks: flattenedCodebooks
        )
    }

    private func selectClosestCentroids(to vector: [Float], limit: Int) -> [(Int, Float)] {
        guard limit > 0 else {
            return []
        }
        guard !flattenedCoarseCentroids.isEmpty else {
            return []
        }

        var closest = BoundedSortedList<(Int, Float)>(capacity: min(limit, coarseCentroids.count)) {
            $0.1 < $1.1
        }

        vector.withUnsafeBufferPointer { vectorBuffer in
            flattenedCoarseCentroids.withUnsafeBufferPointer { centroidBuffer in
                let vectorBase = vectorBuffer.baseAddress!
                let centroidBase = centroidBuffer.baseAddress!
                for index in coarseCentroids.indices {
                    let distance = SIMDDistance.distance(
                        vectorBase,
                        centroidBase.advanced(by: index * dimension),
                        dim: dimension,
                        metric: config.metric
                    )
                    closest.insert((index, distance))
                }
            }
        }
        return closest.elements
    }

    private nonisolated static func nearestCoarseCentroid(
        for vector: [Float],
        flattenedCentroids: [Float],
        centroidCount: Int,
        dimension: Int,
        metric: Metric
    ) -> Int {
        guard centroidCount > 0, !flattenedCentroids.isEmpty else {
            return 0
        }

        var bestIndex = 0
        var bestDistance = Float.greatestFiniteMagnitude
        vector.withUnsafeBufferPointer { vectorBuffer in
            flattenedCentroids.withUnsafeBufferPointer { centroidBuffer in
                guard
                    let vectorBase = vectorBuffer.baseAddress,
                    let centroidBase = centroidBuffer.baseAddress
                else {
                    return
                }
                for index in 0..<centroidCount {
                    let distance = SIMDDistance.distance(
                        vectorBase,
                        centroidBase.advanced(by: index * dimension),
                        dim: dimension,
                        metric: metric
                    )
                    if distance < bestDistance {
                        bestDistance = distance
                        bestIndex = index
                    }
                }
            }
        }
        return bestIndex
    }

    private func restore(from state: PersistedState) throws {
        let rebuiltBuffer = try PQVectorBuffer(capacity: state.capacity, dim: state.dimension, pq: state.pq)
        for (index, code) in state.vectorCodes.enumerated() {
            try rebuiltBuffer.insertEncoded(code: code, at: index)
        }

        self.coarseCentroids = state.coarseCentroids
        self.flattenedCoarseCentroids = state.coarseCentroids.flatMap { $0 }
        self.pq = state.pq
        self.flattenedCodebooks = GPUADCSearch.flattenCodebooks(from: state.pq)
        self.vectorBuffer = rebuiltBuffer
        self.invertedLists = state.invertedLists
        self.coarseAssignments = state.coarseAssignments
        self.idMap = state.idMap
        self.isTrained = true
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 24))
    }

    private static func readUInt32(from data: Data, cursor: inout Int) throws -> UInt32 {
        guard cursor + 4 <= data.count else {
            throw ANNSError.corruptFile("Unexpected EOF in IVFPQ persistence data")
        }
        let b0 = UInt32(data[cursor])
        let b1 = UInt32(data[cursor + 1]) << 8
        let b2 = UInt32(data[cursor + 2]) << 16
        let b3 = UInt32(data[cursor + 3]) << 24
        cursor += 4
        return b0 | b1 | b2 | b3
    }

}

// Synchronized via DispatchQueue serial execution.
private final class ParallelAddPlanningState: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var clusterAssignments: [Int]
    private(set) var encodedCodes: [[UInt8]]
    private(set) var firstError: Error?

    init(vectorCount: Int, codeLength: Int) {
        clusterAssignments = [Int](repeating: 0, count: vectorCount)
        encodedCodes = Array(repeating: [UInt8](repeating: 0, count: codeLength), count: vectorCount)
    }

    var shouldContinue: Bool {
        lock.lock()
        defer { lock.unlock() }
        return firstError == nil
    }

    func store(cluster: Int, code: [UInt8], at index: Int) {
        lock.lock()
        clusterAssignments[index] = cluster
        encodedCodes[index] = code
        lock.unlock()
    }

    func store(error: Error) {
        lock.lock()
        if firstError == nil {
            firstError = error
        }
        lock.unlock()
    }
}
