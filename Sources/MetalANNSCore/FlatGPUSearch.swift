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
package enum FlatGPUSearch {
    package static let maxTopK = 256
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

    package static func search(
        context: MetalContext?,
        query: [Float],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric,
        tierOverride: Int? = nil
    ) async throws -> [SearchResult] {
        // An explicit tier override pins legacy behavior exactly (regression
        // tests rely on this): <=override runs the fp32 host scan, anything
        // larger dispatches the GPU kernel. The int8 tier is bypassed.
        if let tierOverride {
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

        // Default routing.
        // Tier 1: exact int8-bounded prefilter scan for mid-size corpora —
        // reads a quarter of the fp32 bytes while provably preserving the
        // true top-k. Wins roughly in the 16k–45k band where the fp32 host
        // scan is bandwidth-bound but the GPU dispatch tax still dominates.
        let vectorCountForTiering = vectors.count
        if vectorCountForTiering >= BoundedExactScan.minVectorCount,
            vectorCountForTiering <= gpuTierMinVectorCount,
            let bounded = boundedExactSearch(
                query: query, vectors: vectors, k: k, metric: metric
            )
        {
            return bounded
        }
        // Tier 2: parallel fp32 host scan below the dispatch-tax crossover.
        if shouldUseHostPath(vectors: vectors, k: k, tierOverride: nil) || context == nil {
            return hostSearch(query: query, vectors: vectors, k: k, metric: metric)
        }
        // Tier 3: single-dispatch GPU flat scan (bandwidth-bound corpora).
        guard let context else { throw ANNSError.searchFailed("unreachable") }
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
    package static func batchSearch(
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
            return parallelHostBatchSearch(
                queries: queries, vectors: vectors, k: k, metric: metric
            )
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
            return parallelHostBatchSearch(
                queries: queries, vectors: vectors, k: k, metric: metric
            )
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
    package static func isEligible(
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

    /// Above this count the GPU flat scan beats the CPU int8 prefilter tier:
    /// GPU streaming bandwidth overtakes the CPU's DRAM throughput once the
    /// dispatch tax is amortized over a large kernel. Measured crossover on
    /// M3 Max (dim 384): int8 ~319us vs GPU ~420us at 32k; int8 ~900us vs
    /// GPU ~550us at 50k.
    static let gpuTierMinVectorCount = 40960

    private static func shouldUseHostPath(
        vectors: any VectorStorage,
        k: Int,
        tierOverride: Int?
    ) -> Bool {
        vectors.count <= (tierOverride ?? hostTierMaxVectorCount)
    }

    /// Above this vector count a single-query GPU-tier selection fans out
    /// across cores; below it the heap scan is cheaper than lane dispatch.
    static let parallelSelectionMinVectorCount = 16384

    /// Clears cached host-tier corpus norms. Called by mutable storage on
    /// in-place writes so cached norms can never go stale.
    package static func invalidateHostNormCache(buffer: MTLBuffer) {
        normCache.invalidate(bufferID: ObjectIdentifier(buffer))
    }

    package static func invalidateHostNormCache() {
        normCache.clear()
    }

    private static let normCache = CorpusNormCache()

    private struct CorpusNormKey: Hashable {
        let bufferID: ObjectIdentifier
        let bufferLength: Int
        let count: Int
        let dim: Int
    }

    /// Immutable boxed norms array. Publication replaces the dictionary
    /// reference atomically; in-flight readers retain their snapshot, so
    /// scans and top-K selection run entirely without holding the cache
    /// lock (holding it serialized every concurrent host-tier search).
    private final class CachedCorpusNorms {
        let pointer: UnsafeMutablePointer<Float>
        let count: Int

        init(count: Int) {
            let capacity = max(count, 1)
            self.pointer = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
            self.count = count
        }

        deinit {
            if count > 0 {
                pointer.deinitialize(count: count)
            }
            pointer.deallocate()
        }
    }

    // Small fixed-capacity LRU synchronized via NSLock; the lock guards
    // dictionary access only, never downstream computation.
    private final class CorpusNormCache: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [CorpusNormKey: CachedCorpusNorms] = [:]
        private var order: [CorpusNormKey] = []
        private let capacity = 16

        func get(_ key: CorpusNormKey) -> CachedCorpusNorms? {
            lock.lock()
            defer { lock.unlock() }
            return entries[key]
        }

        func store(_ norms: CachedCorpusNorms, for key: CorpusNormKey) {
            lock.lock()
            defer { lock.unlock() }
            if entries[key] == nil {
                order.append(key)
                while order.count > capacity {
                    entries[order.removeFirst()] = nil
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
            }
            order.removeAll { $0.bufferID == bufferID }
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
    package nonisolated static func hostSearch(
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

    /// Exact scan accelerated by the int8 bounded prefilter.
    ///
    /// Returns nil whenever the tier does not apply (storage ineligible,
    /// corpus below threshold) so callers fall back to the plain fp32
    /// parallel scan. Results are brute-force-exact on all paths.
    static func boundedExactSearch(
        query: [Float],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric
    ) -> [SearchResult]? {
        guard let vectorBuffer = vectors as? VectorBuffer,
            !vectors.isFloat16,
            let corpus = vectorBuffer.floatPointer.baseAddress
        else { return nil }
        let vectorCount = vectors.count
        let dim = vectors.dim
        guard vectorCount >= BoundedExactScan.minVectorCount, dim > 0,
            query.count == dim, k > 0, metric != .hamming
        else { return nil }
        guard
            let codeBuffer = Int8CodeCache.codes(
                for: vectorBuffer.buffer,
                vectorCount: vectorCount,
                dim: dim,
                corpus: corpus
            )
        else { return nil }

        var queryNormSq: Float = 0
        if metric == .cosine || metric == .l2 {
            for value in query { queryNormSq += value * value }
        }

        var entries: [ParallelFlatScan.Entry]?
        let cacheKey = CorpusNormKey(
            bufferID: ObjectIdentifier(vectorBuffer.buffer),
            bufferLength: vectorBuffer.buffer.length,
            count: vectorCount,
            dim: dim
        )
        if metric == .innerProduct {
            entries = query.withUnsafeBufferPointer { queryBuffer in
                BoundedExactScan.search(
                    query: queryBuffer.baseAddress!,
                    corpus: corpus,
                    codes: codeBuffer,
                    norms: nil,
                    queryNormSq: 0,
                    vectorCount: vectorCount,
                    dim: dim,
                    topK: min(k, maxTopK),
                    metric: metric
                )
            }
        } else {
            withCorpusSquaredNorms(
                corpus: corpus, vectorCount: vectorCount, dim: dim, key: cacheKey
            ) { normsBase in
                entries = query.withUnsafeBufferPointer { queryBuffer in
                    BoundedExactScan.search(
                        query: queryBuffer.baseAddress!,
                        corpus: corpus,
                        codes: codeBuffer,
                        norms: normsBase,
                        queryNormSq: queryNormSq,
                        vectorCount: vectorCount,
                        dim: dim,
                        topK: min(k, maxTopK),
                        metric: metric
                    )
                }
            }
        }
        guard let result = entries else { return nil }
        return result.map { entry in
            SearchResult(id: "", score: entry.distance, internalID: entry.id)
        }
    }

    /// Row-wise squared norms of the corpus (cached per buffer+shape).
    /// Streams the cached array through `body` without copying. The cache
    /// lock is never held while `body` runs.
    private static func withCorpusSquaredNorms(
        corpus: UnsafePointer<Float>,
        vectorCount: Int,
        dim: Int,
        key: CorpusNormKey?,
        _ body: (UnsafePointer<Float>) -> Void
    ) {
        if let key, let cached = normCache.get(key), cached.count >= vectorCount {
            body(UnsafePointer(cached.pointer))
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

        guard let key else {
            norms.withUnsafeBufferPointer { body($0.baseAddress!) }
            return
        }

        let box = CachedCorpusNorms(count: vectorCount)
        norms.withUnsafeBufferPointer { source in
            box.pointer.initialize(from: source.baseAddress!, count: vectorCount)
        }
        normCache.store(box, for: key)
        // Hand out whichever entry is canonical now (a racing writer may have
        // published its own); every published box holds valid norms for the
        // buffer state it was computed over.
        let published = normCache.get(key) ?? box
        body(UnsafePointer(published.pointer))
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
        var queryNormSq: Float = 0
        if metric == .cosine || metric == .l2 {
            for value in query { queryNormSq += value * value }
        }

        var entries: [ParallelFlatScan.Entry] = []

        func runScan(normsBase: UnsafePointer<Float>?) {
            query.withUnsafeBufferPointer { queryBuffer in
                guard let qBase = queryBuffer.baseAddress else { return }
                entries = ParallelFlatScan.search(
                    query: qBase,
                    corpus: corpus,
                    norms: normsBase,
                    queryNormSq: queryNormSq,
                    vectorCount: vectorCount,
                    dim: dim,
                    topK: topK,
                    metric: metric
                )
            }
        }

        switch metric {
        case .cosine, .l2:
            withCorpusSquaredNorms(
                corpus: corpus, vectorCount: vectorCount, dim: dim, key: cacheKey
            ) { normsBase in
                runScan(normsBase: normsBase)
            }
        case .innerProduct:
            runScan(normsBase: nil)
        case .hamming:
            break
        }

        return entries.map { entry in
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

    /// Runs every query's host-tier scan concurrently across cores. Queries
    /// are independent; results land in preallocated slots so no locking is
    /// needed beyond the caches' internal locks.
    private static func parallelHostBatchSearch(
        queries: [[Float]],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric
    ) -> [[SearchResult]] {
        let count = queries.count
        guard count > 1 else {
            return queries.map { hostSearch(query: $0, vectors: vectors, k: k, metric: metric) }
        }

        let storage = UnsafeMutableBufferPointer<[SearchResult]>.allocate(capacity: count)
        storage.initialize(repeating: [])
        defer {
            storage.deinitialize()
            storage.deallocate()
        }

        let box = HostBatchWork(
            queries: queries, vectors: vectors, k: k, metric: metric, results: storage
        )
        DispatchQueue.concurrentPerform(iterations: count) { index in
            box.run(index)
        }
        return Array(storage)
    }

    /// Bounded max-heap top-K over each query's distance row.
    /// Batched queries select in parallel (rows are independent); a single
    /// large scan is split into per-core chunks whose bounded heaps are then
    /// merged deterministically by (distance, id).
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
        let concurrentBasePointer = ConcurrentReadPointer(pointer: UnsafePointer(basePointer))
        let resultK = min(topK, vectorCount)

        var results = [[SearchResult]](repeating: [], count: queryCount)

        if queryCount > 1 {
            // Rows are independent; one lane per query. Writes target
            // distinct slots through a unique buffer pointer (no COW race).
            results.withUnsafeMutableBufferPointer { buffer in
                let slot = ConcurrentSlotBuffer(buffer: buffer)
                DispatchQueue.concurrentPerform(iterations: queryCount) { queryIndex in
                    slot[queryIndex] = selectRow(
                        concurrentBasePointer.advanced(by: queryIndex * vectorCount),
                        count: vectorCount,
                        resultK: resultK
                    )
                }
            }
            return results
        }

        let activeCores = ProcessInfo.processInfo.activeProcessorCount
        if vectorCount >= parallelSelectionMinVectorCount && activeCores > 1 {
            let lanes = min(activeCores, vectorCount / parallelSelectionMinVectorCount)
            if lanes > 1 {
                results[0] = selectRowParallel(
                    concurrentBasePointer.pointer,
                    count: vectorCount,
                    resultK: resultK,
                    lanes: lanes
                )
                return results
            }
        }

        results[0] = selectRow(concurrentBasePointer.pointer, count: vectorCount, resultK: resultK)
        return results
    }

    private struct CandidateEntry {
        var distance: Float
        var id: UInt32
    }

    /// Bounded max-heap top-K over `count` consecutive floats at `row`.
    /// Candidate ids are `idBase + index`; chunked callers must pass their
    /// chunk's start offset so ids stay global row indices.
    /// Returns entries sorted ascending by (distance, id).
    private static func boundedTopK(
        _ row: UnsafePointer<Float>,
        count: Int,
        topK: Int,
        idBase: UInt32 = 0
    ) -> [CandidateEntry] {
        guard count > 0, topK > 0 else { return [] }
        var heapDistances = ContiguousArray<Float>()
        var heapIDs = ContiguousArray<UInt32>()
        heapDistances.reserveCapacity(topK)
        heapIDs.reserveCapacity(topK)

        @inline(__always) func siftDown(from index: Int, size: Int) {
            var position = index
            while true {
                let left = 2 * position + 1
                let right = left + 1
                var largest = position
                if left < size, heapDistances[left] > heapDistances[largest] {
                    largest = left
                }
                if right < size, heapDistances[right] > heapDistances[largest] {
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

        for index in 0..<count {
            let value = row[index]
            if heapDistances.count < topK {
                heapDistances.append(value)
                heapIDs.append(UInt32(index) &+ idBase)
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
                heapIDs[0] = UInt32(index) &+ idBase
                siftDown(from: 0, size: topK)
            }
        }

        // Extract ascending by repeatedly swapping the max to the end.
        var entries = [CandidateEntry]()
        entries.reserveCapacity(min(topK, heapDistances.count))
        var size = heapDistances.count
        while size > 0 {
            entries.append(CandidateEntry(distance: heapDistances[0], id: heapIDs[0]))
            size -= 1
            if size > 0 {
                heapDistances[0] = heapDistances[size]
                heapIDs[0] = heapIDs[size]
                siftDown(from: 0, size: size)
            }
        }
        // Root-first extraction of a max-heap yields descending distances;
        // flip to ascending (nearest-first) for output.
        entries.reverse()

        // Make ties deterministic (only pay the sort when duplicate
        // distances actually exist).
        var hasTies = false
        if entries.count > 1 {
            var index = 1
            while index < entries.count {
                if entries[index].distance == entries[index - 1].distance {
                    hasTies = true
                    break
                }
                index += 1
            }
        }
        if hasTies {
            entries.sort { $0.distance == $1.distance ? $0.id < $1.id : $0.distance < $1.distance }
        }
        return entries
    }

    private static func selectRow(
        _ row: UnsafePointer<Float>,
        count: Int,
        resultK: Int
    ) -> [SearchResult] {
        boundedTopK(row, count: count, topK: resultK).map { entry in
            SearchResult(id: "", score: entry.distance, internalID: entry.id)
        }
    }

    /// Splits one long distance row into `lanes` chunks, selects per-chunk
    /// top-K concurrently, then merges by (distance, id). Result ordering is
    /// identical to the serial path up to tie order among equal distances.
    private static func selectRowParallel(
        _ row: UnsafePointer<Float>,
        count: Int,
        resultK: Int,
        lanes: Int
    ) -> [SearchResult] {
        let chunkSize = count / lanes
        let concurrentRow = ConcurrentReadPointer(pointer: row)
        var laneEntries = [[CandidateEntry]](repeating: [], count: lanes)
        laneEntries.withUnsafeMutableBufferPointer { buffer in
            let slots = ConcurrentSlotBuffer(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: lanes) { lane in
                let start = lane * chunkSize
                let end = lane == lanes - 1 ? count : start + chunkSize
                slots[lane] = boundedTopK(
                    concurrentRow.advanced(by: start),
                    count: end - start,
                    topK: resultK,
                    idBase: UInt32(start)
                )
            }
        }

        var merged: [CandidateEntry] = laneEntries.flatMap { $0 }
        merged.sort { $0.distance == $1.distance ? $0.id < $1.id : $0.distance < $1.distance }
        return merged.prefix(resultK).map { entry in
            SearchResult(id: "", score: entry.distance, internalID: entry.id)
        }
    }
}

/// Unchecked-Sendable read-only view over a pointer whose owner keeps the
/// backing storage alive for the synchronous `concurrentPerform` scope.
private struct ConcurrentReadPointer<Element>: @unchecked Sendable {
    let pointer: UnsafePointer<Element>

    func advanced(by offset: Int) -> UnsafePointer<Element> {
        pointer.advanced(by: offset)
    }
}

/// Unchecked-Sendable view over a uniquely-owned buffer so concurrent
/// lanes can write distinct indices without tripping COW data races.
private struct ConcurrentSlotBuffer<Element>: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<Element>

    subscript(_ index: Int) -> Element {
        get { buffer[index] }
        nonmutating set { buffer[index] = newValue }
    }
}

/// Scoped sendable box for parallel per-query work; slots are disjoint and
/// the underlying buffers outlive the concurrentPerform join.
private final class HostBatchWork: @unchecked Sendable {
    private let queries: [[Float]]
    private let vectors: any VectorStorage
    private let k: Int
    private let metric: Metric
    private let results: UnsafeMutableBufferPointer<[SearchResult]>

    init(
        queries: [[Float]],
        vectors: any VectorStorage,
        k: Int,
        metric: Metric,
        results: UnsafeMutableBufferPointer<[SearchResult]>
    ) {
        self.queries = queries
        self.vectors = vectors
        self.k = k
        self.metric = metric
        self.results = results
    }

    func run(_ index: Int) {
        // Mid-band corpora route through the exact int8 prefilter, mirroring
        // single-query tier selection; everything else uses the fp32 scan.
        if let bounded = FlatGPUSearch.boundedExactSearch(
            query: queries[index], vectors: vectors, k: k, metric: metric
        ) {
            results[index] = bounded
        } else {
            results[index] = FlatGPUSearch.hostSearch(
                query: queries[index], vectors: vectors, k: k, metric: metric
            )
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
            throw ANNSError.gpuPipelineUnavailable("Metal function '\(name)' not found")
        }
        let state = try device.makeComputePipelineState(function: function)

        lock.lock()
        pipelines[deviceKey, default: [:]][name] = state
        lock.unlock()
        return state
    }
}

// Synchronized via NSLock; safe for concurrent acquire/release.
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
