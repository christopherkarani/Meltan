import Accelerate
import Foundation
import Metal
import simd

/// Single-dispatch exact nearest-neighbor search over the full vector corpus.
///
/// Replaces the hybrid beam-search loop (one command-buffer commit + sync per
/// graph hop) with ONE bandwidth-bound distance-scan kernel executed in a
/// single command buffer, followed by host-side bounded-heap top-K selection.
/// Results are exact: recall@k == 1.0 against brute force.
public enum FlatGPUSearch {
    public static let maxTopK = 256
    static let tileRows = 2048
    static let minEligibleVectorCount = 64
    /// Upper bound on scratch bytes per dispatch chunk (~64 MiB).
    private static let maxScratchBytesPerDispatch = 64 * 1024 * 1024

    private static let emptyID: UInt32 = 0xFFFF_FFFF

    private static let workspacePool = FlatWorkspacePool()
    private static let pipelineStore = FlatPipelineStore()

    private struct Pair {
        var distance: Float
        var id: UInt32
    }

    // MARK: - Public API

    public static func search(
        context: MetalContext?,
        query: [Float],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric,
        tierOverride: Int? = nil
    ) async throws -> [SearchResult] {
        if shouldUseHostPath(vectors: vectors, k: k, tierOverride: tierOverride) {
            return hostSearch(query: query, vectors: vectors, k: k, metric: metric)
        }
        guard let context else {
            return hostSearch(query: query, vectors: vectors, k: k, metric: metric)
        }
        let batches = try await batchSearch(
            context: context,
            queries: [query],
            vectors: vectors,
            k: k,
            metric: metric,
            tierOverride: tierOverride
        )
        guard let first = batches.first else {
            throw ANNSError.searchFailed("FlatGPUSearch produced no results")
        }
        return first
    }

    /// Exact batched nearest-neighbor search.
    ///
    /// The GPU tier reads `vectors.buffer` as row-major Float32; callers must
    /// only pass storage with that layout (e.g. via `isEligible`). Storage with
    /// other layouts falls back to the correct host scan below
    /// `hostTierMaxVectorCount` and should not be routed here above it.
    public static func batchSearch(
        context: MetalContext?,
        queries: [[Float]],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric,
        tierOverride: Int? = nil
    ) async throws -> [[SearchResult]] {
        guard !queries.isEmpty else { return [] }
        guard k > 0 else { return Array(repeating: [], count: queries.count) }

        let vectorCount = vectors.count
        guard vectorCount > 0 else {
            return Array(repeating: [], count: queries.count)
        }
        guard metric != .hamming else {
            throw ANNSError.searchFailed("FlatGPUSearch does not support metric .hamming")
        }

        let dim = vectors.dim
        for query in queries where query.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }

        if shouldUseHostPath(vectors: vectors, k: k, tierOverride: tierOverride) || context == nil {
            return queries.map { hostSearch(query: $0, vectors: vectors, k: k, metric: metric) }
        }

        let effectiveK = min(k, maxTopK, vectorCount)
        let metricType: UInt32 =
            switch metric {
            case .cosine: 0
            case .l2: 1
            case .innerProduct: 2
            case .hamming: 3
            }

        guard let context else {
            return queries.map { hostSearch(query: $0, vectors: vectors, k: k, metric: metric) }
        }

        let scanPipeline = try pipeline(named: "flat_scan_distances", context: context)

        let maxChunkQueries = max(
            1,
            min(
                queries.count,
                maxQueriesPerDispatch(vectorCount: vectorCount),
                Int(UInt32.max) / max(tileRows, 1)
            )
        )

        var allResults = [[SearchResult]]()
        allResults.reserveCapacity(queries.count)

        var chunkStart = 0
        while chunkStart < queries.count {
            let chunkEnd = min(chunkStart + maxChunkQueries, queries.count)
            let chunk = queries[chunkStart..<chunkEnd]
            let chunkResults = try await runChunk(
                context: context,
                pipeline: scanPipeline,
                queries: Array(chunk),
                vectors: vectors,
                vectorCount: vectorCount,
                dim: dim,
                topK: effectiveK,
                metricType: metricType
            )
            allResults.append(contentsOf: chunkResults)
            chunkStart = chunkEnd
        }

        return allResults
    }

    // MARK: - Host tier (small corpora, no GPU dispatch)

    /// Eligibility gate for the flat exact-search path.
    ///
    /// The GPU kernel indexes raw Float32 rows in `vectors.buffer`, so only
    /// storage with that layout qualifies: `VectorBuffer` (fully resident,
    /// row-major Float32) and Float32-mode `MmapVectorStorage` (zero-copy
    /// file pages wrapped in a shared MTLBuffer — identical row layout).
    /// Binary-packed, Float16, and disk-backed staging windows are excluded
    /// structurally rather than by per-type flags.
    public static func isEligible(
        vectors: any VectorStorage,
        metric: Metric,
        k: Int,
        maxVectorCount: Int
    ) -> Bool {
        guard vectors is VectorBuffer || isEligibleMmapStorage(vectors) else { return false }
        guard metric != .hamming else { return false }
        guard k >= 1, k <= maxTopK else { return false }
        guard vectors.count >= minEligibleVectorCount else { return false }
        guard maxVectorCount > 0, vectors.count <= maxVectorCount else { return false }
        return true
    }

    private static func isEligibleMmapStorage(_ vectors: any VectorStorage) -> Bool {
        (vectors as? MmapVectorStorage)?.isFlatScanEligible == true
    }

    /// Below this vector count the host BLAS scan beats a GPU dispatch round
    /// trip on Apple Silicon (a single dispatch costs tens to hundreds of
    /// microseconds depending on power state). Measured crossover on M3 Max:
    /// host scan streams at ~60-90 GB/s single-core while the GPU flat path
    /// pays a fixed submission tax then reads at ~150-200 GB/s.
    static let hostTierMaxVectorCount = 32768

    private static func shouldUseHostPath(
        vectors: any VectorStorage,
        k: Int,
        tierOverride: Int?
    ) -> Bool {
        vectors.count <= (tierOverride ?? hostTierMaxVectorCount)
    }

    /// Clears cached host-tier corpus norms. Called by mutable storage on
    /// in-place writes so cached norms can never go stale.
    public static func invalidateHostNormCache(buffer: MTLBuffer) {
        normCache.invalidate(bufferID: ObjectIdentifier(buffer))
    }

    public static func invalidateHostNormCache() {
        normCache.clear()
    }

    private static let normCache = CorpusNormCache()

    private struct CorpusNormKey: Hashable {
        let bufferID: ObjectIdentifier
        let bufferLength: Int
        let count: Int
        let dim: Int
    }

    // Synchronized via NSLock; small fixed-capacity LRU.
    private final class CorpusNormCache: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [CorpusNormKey: [Float]] = [:]
        private var order: [CorpusNormKey] = []
        private let capacity = 16

        func norms(for key: CorpusNormKey) -> [Float]? {
            lock.lock()
            defer { lock.unlock() }
            return entries[key]
        }

        func withNorms(for key: CorpusNormKey, _ body: (UnsafePointer<Float>) -> Void) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard let norms = entries[key], let base = norms.withUnsafeBufferPointer({ $0.baseAddress }) else {
                return false
            }
            body(base)
            return true
        }

        func store(_ norms: [Float], for key: CorpusNormKey) {
            lock.lock()
            defer { lock.unlock() }
            if entries[key] == nil {
                order.append(key)
                while order.count > capacity {
                    let evicted = order.removeFirst()
                    entries[evicted] = nil
                }
            }
            entries[key] = norms
        }

        func invalidate(bufferID: ObjectIdentifier) {
            lock.lock()
            defer { lock.unlock() }
            let doomed = order.filter { $0.bufferID == bufferID }
            for key in doomed {
                entries[key] = nil
                order.removeAll { $0.bufferID == bufferID }
            }
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            entries.removeAll()
            order.removeAll()
        }
    }

    /// Fused brute-force scan over the raw corpus pointer with a bounded
    /// max-heap top-K. Avoids any Metal dispatch overhead; used when the
    /// corpus is small enough that a GPU round trip would dominate.
    nonisolated static func hostSearch(
        query: [Float],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric
    ) -> [SearchResult] {
        let vectorCount = vectors.count
        guard vectorCount > 0, k > 0 else { return [] }
        let dim = vectors.dim
        guard query.count == dim else {
            return []
        }
        let effectiveK = min(k, maxTopK, vectorCount)

        // Fast path: zero-copy access to the contiguous Float32 buffer.
        if let vectorBuffer = vectors as? VectorBuffer,
            !vectors.isFloat16,
            let raw = vectorBuffer.floatPointer.baseAddress
        {
            return hostScan(
                query: query,
                corpus: raw,
                vectorCount: vectorCount,
                dim: dim,
                topK: effectiveK,
                metric: metric,
                cacheKey: CorpusNormKey(
                    bufferID: ObjectIdentifier(vectorBuffer.buffer),
                    bufferLength: vectorBuffer.buffer.length,
                    count: vectorCount,
                    dim: dim
                )
            )
        }

        // Generic fallback: materialize rows once (correctness over speed).
        var flat = [Float](repeating: 0, count: vectorCount * dim)
        flat.withUnsafeMutableBufferPointer { destination in
            for row in 0..<vectorCount {
                let vector = vectors.vector(at: row)
                destination.baseAddress!.advanced(by: row * dim).update(from: vector, count: dim)
            }
        }
        return flat.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return [] }
            return hostScan(
                query: query,
                corpus: base,
                vectorCount: vectorCount,
                dim: dim,
                topK: effectiveK,
                metric: metric,
                cacheKey: nil
            )
        }
    }

    /// Row-wise dot products of every corpus row against `query` via BLAS.
    private static func corpusDotProducts(
        corpus: UnsafePointer<Float>,
        query: UnsafePointer<Float>,
        vectorCount: Int,
        dim: Int,
        output: UnsafeMutablePointer<Float>
    ) {
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            Int32(vectorCount),
            Int32(dim),
            1.0,
            corpus,
            Int32(dim),
            query,
            1,
            0.0,
            output,
            1
        )
    }

    /// Row-wise squared norms of the corpus (cached per buffer+shape).
    /// Streams the cached array through `body` without copying.
    private static func withCorpusSquaredNorms(
        corpus: UnsafePointer<Float>,
        vectorCount: Int,
        dim: Int,
        key: CorpusNormKey?,
        _ body: (UnsafePointer<Float>) -> Void
    ) {
        if let key, normCache.withNorms(for: key, body) {
            return
        }

        var norms = [Float](repeating: 0, count: vectorCount)
        norms.withUnsafeMutableBufferPointer { normsBuffer in
            guard let normsBase = normsBuffer.baseAddress else { return }
            let elementCount = vectorCount * dim
            var squared = [Float](repeating: 0, count: elementCount)
            squared.withUnsafeMutableBufferPointer { squaredBuffer in
                guard let squaredBase = squaredBuffer.baseAddress else { return }
                vDSP_vsq(corpus, 1, squaredBase, 1, vDSP_Length(elementCount))
                let ones = [Float](repeating: 1.0, count: dim)
                ones.withUnsafeBufferPointer { onesBuffer in
                    guard let onesBase = onesBuffer.baseAddress else { return }
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(vectorCount),
                        Int32(dim),
                        1.0,
                        squaredBase,
                        Int32(dim),
                        onesBase,
                        1,
                        0.0,
                        normsBase,
                        1
                    )
                }
            }
        }

        if let key {
            normCache.store(norms, for: key)
            _ = normCache.withNorms(for: key, body)
        } else {
            norms.withUnsafeBufferPointer { body($0.baseAddress!) }
        }
    }

    private static func hostScan(
        query: [Float],
        corpus: UnsafePointer<Float>,
        vectorCount: Int,
        dim: Int,
        topK: Int,
        metric: Metric,
        cacheKey: CorpusNormKey?
    ) -> [SearchResult] {
        var dots = [Float](repeating: 0, count: vectorCount)

        query.withUnsafeBufferPointer { queryBuffer in
            guard let qBase = queryBuffer.baseAddress else { return }
            dots.withUnsafeMutableBufferPointer { dotsBuffer in
                guard let dotsBase = dotsBuffer.baseAddress else { return }
                corpusDotProducts(
                    corpus: corpus,
                    query: qBase,
                    vectorCount: vectorCount,
                    dim: dim,
                    output: dotsBase
                )
            }
        }

        var queryNormSq: Float = 0
        if metric == .cosine || metric == .l2 {
            for value in query { queryNormSq += value * value }
        }

        var selector = TopKSelector(topK: topK)
        dots.withUnsafeBufferPointer { dotsBuffer in
            guard let dotsBase = dotsBuffer.baseAddress else { return }

            switch metric {
            case .cosine:
                withCorpusSquaredNorms(
                    corpus: corpus, vectorCount: vectorCount, dim: dim, key: cacheKey
                ) { normsBase in
                    for row in 0..<vectorCount {
                        let denom = (queryNormSq * normsBase[row]).squareRoot()
                        let distance = denom < 1e-10 ? 1.0 : (1.0 - dotsBase[row] / denom)
                        selector.append(distance: distance, id: UInt32(row))
                    }
                }
            case .l2:
                withCorpusSquaredNorms(
                    corpus: corpus, vectorCount: vectorCount, dim: dim, key: cacheKey
                ) { normsBase in
                    for row in 0..<vectorCount {
                        let distance = max(0, queryNormSq - 2.0 * dotsBase[row] + normsBase[row])
                        selector.append(distance: distance, id: UInt32(row))
                    }
                }
            case .innerProduct:
                for row in 0..<vectorCount {
                    selector.append(distance: -dotsBase[row], id: UInt32(row))
                }
            case .hamming:
                break
            }
        }

        return selector.sortedAscending().map { entry in
            SearchResult(id: "", score: entry.distance, internalID: entry.id)
        }
    }

    /// Bounded max-heap over (distance, id) pairs; append() is O(1) amortized
    /// once full, sortedAscending() extracts entries in ascending order.
    private struct TopKSelector {
        struct Entry {
            var distance: Float
            var id: UInt32
        }

        private var entries: [Entry]
        private var size = 0

        init(topK: Int) {
            let capacity = max(topK, 1)
            entries = [Entry](repeating: Entry(distance: 0, id: 0), count: capacity)
        }

        @inline(__always) mutating func append(distance: Float, id: UInt32) {
            if size < entries.count {
                entries[size] = Entry(distance: distance, id: id)
                var position = size
                size += 1
                while position > 0 {
                    let parent = (position - 1) / 2
                    if entries[position].distance > entries[parent].distance {
                        entries.swapAt(position, parent)
                        position = parent
                    } else {
                        break
                    }
                }
            } else if distance < entries[0].distance {
                entries[0] = Entry(distance: distance, id: id)
                siftDown(from: 0, size: size)
            }
        }

        private mutating func siftDown(from start: Int, size: Int) {
            var position = start
            while true {
                let left = 2 &* position &+ 1
                let right = left &+ 1
                var largest = position
                if left < size, entries[left].distance > entries[largest].distance { largest = left }
                if right < size, entries[right].distance > entries[largest].distance { largest = right }
                if largest == position { return }
                entries.swapAt(position, largest)
                position = largest
            }
        }

        func sortedAscending() -> [Entry] {
            var heap = entries
            var count = min(size, heap.count)
            while count > 1 {
                heap.swapAt(0, count - 1)
                count -= 1
                var position = 0
                while true {
                    let left = 2 &* position &+ 1
                    let right = left &+ 1
                    var largest = position
                    if left < count, heap[left].distance > heap[largest].distance { largest = left }
                    if right < count, heap[right].distance > heap[largest].distance { largest = right }
                    if largest == position { break }
                    heap.swapAt(position, largest)
                    position = largest
                }
            }
            return Array(heap.prefix(min(size, heap.count)))
        }
    }

    // MARK: - Dispatch

    private static func maxQueriesPerDispatch(vectorCount: Int) -> Int {
        let bytesPerQuery = max(vectorCount, 1) * MemoryLayout<Float>.stride
        return max(1, maxScratchBytesPerDispatch / bytesPerQuery)
    }

    private static func runChunk(
        context: MetalContext,
        pipeline: MTLComputePipelineState,
        queries: [[Float]],
        vectors: any VectorStorage,
        vectorCount: Int,
        dim: Int,
        topK: Int,
        metricType: UInt32
    ) async throws -> [[SearchResult]] {
        let queryFloats = queries.count * dim
        let workspace = try workspacePool.acquire(
            device: context.device,
            queryFloats: queryFloats,
            queryCount: queries.count,
            distanceFloats: queries.count * vectorCount
        )
        defer { workspacePool.release(workspace) }

        writeQueries(queries, into: workspace.queryBuffer)
        writeQueryNorms(queries, into: workspace.normBuffer)

        var queryCountValue = UInt32(queries.count)
        var vectorCountValue = UInt32(vectorCount)
        var dimValue = UInt32(dim)
        var metricTypeValue = metricType

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.searchFailed("Failed to create compute command encoder")
            }
            defer { encoder.endEncoding() }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(workspace.queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 1)
            encoder.setBuffer(workspace.normBuffer, offset: 0, index: 2)
            encoder.setBuffer(workspace.distanceBuffer, offset: 0, index: 3)
            encoder.setBytes(&queryCountValue, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&vectorCountValue, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&dimValue, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&metricTypeValue, length: MemoryLayout<UInt32>.stride, index: 7)

            // One thread per (query, row).
            let threadsPerGrid = MTLSize(width: queries.count * vectorCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(width: sanitizedThreadgroupSize(pipeline), height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        }

        return selectTopK(
            distances: workspace.distanceBuffer,
            queryCount: queries.count,
            vectorCount: vectorCount,
            topK: topK
        )
    }

    // MARK: - Host-side helpers

    private static func pipeline(
        named name: String,
        context: MetalContext
    ) throws -> MTLComputePipelineState {
        try pipelineStore.pipeline(named: name, device: context.device, library: context.library)
    }

    private static func sanitizedThreadgroupSize(_ pipeline: MTLComputePipelineState) -> Int {
        let maximum = pipeline.maxTotalThreadsPerThreadgroup
        var size = min(256, maximum)
        size -= size % 32
        return max(size, 32)
    }

    private static func writeQueries(_ queries: [[Float]], into buffer: MTLBuffer) {
        let pointer = buffer.contents().bindMemory(
            to: Float.self,
            capacity: max(buffer.length / MemoryLayout<Float>.stride, 1)
        )
        var offset = 0
        for query in queries {
            query.withUnsafeBufferPointer { source in
                guard let sourceBase = source.baseAddress else { return }
                pointer.advanced(by: offset).update(from: sourceBase, count: source.count)
            }
            offset += query.count
        }
    }

    private static func writeQueryNorms(_ queries: [[Float]], into buffer: MTLBuffer) {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: max(queries.count, 1))
        for (index, query) in queries.enumerated() {
            var sum: Float = 0
            for value in query {
                sum += value * value
            }
            pointer[index] = sum
        }
    }

    /// Bounded max-heap top-K over each query's distance row.
    /// Cost ≈ vectorCount comparisons + O(K·ln(N/K)) replacements.
    private static func selectTopK(
        distances: MTLBuffer,
        queryCount: Int,
        vectorCount: Int,
        topK: Int
    ) -> [[SearchResult]] {
        let basePointer = distances.contents().bindMemory(
            to: Float.self,
            capacity: max(queryCount * vectorCount, 1)
        )

        return (0..<queryCount).map { queryIndex in
            let row = basePointer.advanced(by: queryIndex * vectorCount)
            let resultK = min(topK, vectorCount)

            var heapDistances = ContiguousArray<Float>()
            var heapIDs = ContiguousArray<UInt32>()
            heapDistances.reserveCapacity(resultK)
            heapIDs.reserveCapacity(resultK)

            @inline(__always) func siftDown(from index: Int) {
                var position = index
                let count = heapDistances.count
                while true {
                    let left = 2 * position + 1
                    let right = left + 1
                    var largest = position
                    if left < count, heapDistances[left] > heapDistances[largest] {
                        largest = left
                    }
                    if right < count, heapDistances[right] > heapDistances[largest] {
                        largest = right
                    }
                    if largest == position {
                        return
                    }
                    heapDistances.swapAt(position, largest)
                    heapIDs.swapAt(position, largest)
                    position = largest
                }
            }

            for rowIndex in 0..<vectorCount {
                let value = row[rowIndex]
                if heapDistances.count < resultK {
                    heapDistances.append(value)
                    heapIDs.append(UInt32(rowIndex))
                    var position = heapDistances.count - 1
                    while position > 0 {
                        let parent = (position - 1) / 2
                        if heapDistances[position] > heapDistances[parent] {
                            heapDistances.swapAt(position, parent)
                            heapIDs.swapAt(position, parent)
                            position = parent
                        } else {
                            break
                        }
                    }
                } else if value < heapDistances[0] {
                    heapDistances[0] = value
                    heapIDs[0] = UInt32(rowIndex)
                    siftDown(from: 0)
                }
            }

            // Extract ascending by repeatedly swapping the max to the end.
            var count = heapDistances.count
            while count > 1 {
                heapDistances.swapAt(0, count - 1)
                heapIDs.swapAt(0, count - 1)
                count -= 1
                var position = 0
                while true {
                    let left = 2 * position + 1
                    let right = left + 1
                    var largest = position
                    if left < count, heapDistances[left] > heapDistances[largest] {
                        largest = left
                    }
                    if right < count, heapDistances[right] > heapDistances[largest] {
                        largest = right
                    }
                    if largest == position {
                        break
                    }
                    heapDistances.swapAt(position, largest)
                    heapIDs.swapAt(position, largest)
                    position = largest
                }
            }

            return (0..<resultK).map { slot in
                SearchResult(
                    id: "",
                    score: heapDistances[slot],
                    internalID: heapIDs[slot]
                )
            }
        }
    }
}

// Synchronized via NSLock; safe for concurrent access across isolation domains.
private final class FlatPipelineStore: @unchecked Sendable {
    private let lock = NSLock()
    private var pipelines: [ObjectIdentifier: [String: MTLComputePipelineState]] = [:]

    func pipeline(
        named name: String,
        device: MTLDevice,
        library: MTLLibrary
    ) throws -> MTLComputePipelineState {
        let deviceKey = ObjectIdentifier(device)
        lock.lock()
        if let cached = pipelines[deviceKey]?[name] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        guard let function = library.makeFunction(name: name) else {
            throw ANNSError.constructionFailed("Metal function '\(name)' not found")
        }
        let state = try device.makeComputePipelineState(function: function)

        lock.lock()
        pipelines[deviceKey, default: [:]][name] = state
        lock.unlock()
        return state
    }
}

// Synchronized via NSLock; pool is safe for concurrent acquire/release.
private final class FlatWorkspacePool: @unchecked Sendable {
    final class Workspace: @unchecked Sendable {
        let deviceID: ObjectIdentifier
        let queryBuffer: MTLBuffer
        let normBuffer: MTLBuffer
        let distanceBuffer: MTLBuffer

        init(
            deviceID: ObjectIdentifier,
            queryBuffer: MTLBuffer,
            normBuffer: MTLBuffer,
            distanceBuffer: MTLBuffer
        ) {
            self.deviceID = deviceID
            self.queryBuffer = queryBuffer
            self.normBuffer = normBuffer
            self.distanceBuffer = distanceBuffer
        }
    }

    private let lock = NSLock()
    private var available: [Workspace] = []

    func acquire(
        device: MTLDevice,
        queryFloats: Int,
        queryCount: Int,
        distanceFloats: Int
    ) throws -> Workspace {
        let deviceID = ObjectIdentifier(device)
        let queryBytes = max(queryFloats * MemoryLayout<Float>.stride, 4)
        let normBytes = max(queryCount * MemoryLayout<Float>.stride, 4)
        let distanceBytes = max(distanceFloats * MemoryLayout<Float>.stride, 4)

        lock.lock()
        if let index = available.firstIndex(where: {
            $0.deviceID == deviceID && $0.queryBuffer.length >= queryBytes && $0.normBuffer.length >= normBytes
                && $0.distanceBuffer.length >= distanceBytes
        }) {
            let workspace = available.remove(at: index)
            lock.unlock()
            return workspace
        }
        lock.unlock()

        guard
            let queryBuffer = device.makeBuffer(length: queryBytes, options: .storageModeShared),
            let normBuffer = device.makeBuffer(length: normBytes, options: .storageModeShared),
            let distanceBuffer = device.makeBuffer(length: distanceBytes, options: .storageModeShared)
        else {
            throw ANNSError.searchFailed("Failed to allocate FlatGPUSearch workspace buffers")
        }

        return Workspace(
            deviceID: deviceID,
            queryBuffer: queryBuffer,
            normBuffer: normBuffer,
            distanceBuffer: distanceBuffer
        )
    }

    func release(_ workspace: Workspace) {
        lock.lock()
        available.append(workspace)
        if available.count > 4 {
            available.removeFirst(available.count - 4)
        }
        lock.unlock()
    }
}
