import Accelerate
import Foundation
import Metal
import MetalANNSC
import os

/// IVF-flat: coarse k-means lists, then an exact fp32 scan of the probed rows.
///
/// When `nprobe >= nlist` every row is a candidate and the result matches
/// brute force (same (distance, id) order as `ParallelFlatScan`). Smaller
/// nprobe is approximate.
public enum IVFFlatSearch {
    public static let minVectorCount = 256
    public static let defaultListCount = 256
    public static let defaultNProbe = 4

    private static let logger = Logger(subsystem: "com.metalanns", category: "IVFFlatSearch")

    /// Default coarse-list count: `min(256, max(16, n/64))`.
    public static func listCount(for vectorCount: Int) -> Int {
        min(defaultListCount, max(16, vectorCount / 64))
    }

    /// `defaultListCount` (256) means "auto": `listCount(for:)`. There is no way
    /// to pin exactly 256 lists on a corpus whose auto count is smaller; pass
    /// 255 or 257 instead. Any other positive value is honored (clamped to `n`).
    public static func resolvedListCount(requested: Int, vectorCount: Int) -> Int {
        let auto = listCount(for: vectorCount)
        if requested == defaultListCount {
            return min(auto, vectorCount)
        }
        return min(max(1, requested), vectorCount)
    }

    /// Built inverted-list snapshot. Immutable after init: search only reads
    /// owned packed/centroid/list storage. `@unchecked Sendable` because those
    /// buffers are not mutated after publish; the cache `Store` lock serializes
    /// insert/evict.
    public final class Partition: @unchecked Sendable {
        let nlist: Int
        let count: Int
        let dim: Int
        let metric: Metric
        let packed: UnsafeMutablePointer<Float>
        let packedNormSq: UnsafeMutablePointer<Float>
        let centroids: UnsafeMutablePointer<Float>
        let centroidNormSq: UnsafeMutablePointer<Float>
        let rowNormSq: UnsafeMutablePointer<Float>
        let listOffsets: [Int]
        let listIDs: UnsafeMutablePointer<UInt32>
        let listIDCount: Int

        fileprivate init(
            nlist: Int,
            count: Int,
            dim: Int,
            metric: Metric,
            packed: UnsafeMutablePointer<Float>,
            packedNormSq: UnsafeMutablePointer<Float>,
            centroids: UnsafeMutablePointer<Float>,
            centroidNormSq: UnsafeMutablePointer<Float>,
            rowNormSq: UnsafeMutablePointer<Float>,
            listOffsets: [Int],
            listIDs: UnsafeMutablePointer<UInt32>,
            listIDCount: Int
        ) {
            self.nlist = nlist
            self.count = count
            self.dim = dim
            self.metric = metric
            self.packed = packed
            self.packedNormSq = packedNormSq
            self.centroids = centroids
            self.centroidNormSq = centroidNormSq
            self.rowNormSq = rowNormSq
            self.listOffsets = listOffsets
            self.listIDs = listIDs
            self.listIDCount = listIDCount
        }

        deinit {
            packed.deinitialize(count: count * dim)
            packed.deallocate()
            packedNormSq.deinitialize(count: count)
            packedNormSq.deallocate()
            centroids.deinitialize(count: nlist * dim)
            centroids.deallocate()
            centroidNormSq.deinitialize(count: nlist)
            centroidNormSq.deallocate()
            rowNormSq.deinitialize(count: count)
            rowNormSq.deallocate()
            listIDs.deinitialize(count: listIDCount)
            listIDs.deallocate()
        }
    }

    public static func build(
        corpus: UnsafePointer<Float>,
        vectorCount: Int,
        dim: Int,
        nlist: Int,
        metric: Metric,
        seed: UInt64 = 42
    ) throws -> Partition {
        try buildPartition(
            corpus: corpus,
            count: vectorCount,
            dim: dim,
            metric: metric,
            nlist: nlist,
            seed: seed
        )
    }

    public static func search(
        query: [Float],
        partition: Partition,
        k: Int,
        nprobe: Int,
        metric: Metric
    ) throws -> [SearchResult] {
        guard k > 0, partition.count > 0 else { return [] }
        guard query.count == partition.dim else {
            throw ANNSError.dimensionMismatch(expected: partition.dim, got: query.count)
        }
        let effectiveMetric = metric
        return query.withUnsafeBufferPointer { queryBuffer in
            guard let queryBase = queryBuffer.baseAddress else { return [] }
            return scanProbedLists(
                query: queryBase,
                partition: partition,
                k: min(k, partition.count),
                nprobe: max(1, min(nprobe, partition.nlist)),
                metric: effectiveMetric
            )
        }
    }

    public static func search(
        query: [Float],
        vectors: VectorBuffer,
        k: Int,
        nlist: Int,
        nprobe: Int,
        metric: Metric
    ) throws -> [SearchResult] {
        guard k > 0 else { return [] }
        guard metric != .hamming else {
            throw ANNSError.searchFailed("IVF-flat does not support metric .hamming")
        }
        let vectorCount = vectors.count
        let dim = vectors.dim
        guard vectorCount > 0, dim > 0 else { return [] }
        guard query.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }
        guard let corpus = vectors.floatPointer.baseAddress else { return [] }

        let lists = resolvedListCount(requested: nlist, vectorCount: vectorCount)
        let probe = max(1, min(nprobe, lists))
        if probe >= lists {
            return FlatGPUSearch.hostSearch(
                query: query, vectors: vectors, k: k, metric: metric
            )
        }

        let partition: Partition
        do {
            partition = try cachedPartition(
                buffer: vectors.buffer,
                corpus: corpus,
                vectorCount: vectorCount,
                dim: dim,
                metric: metric,
                nlist: lists
            )
        } catch {
            logger.error(
                "IVF-flat partition build failed; falling back to exact scan: \(error.localizedDescription, privacy: .public)"
            )
            return FlatGPUSearch.hostSearch(
                query: query, vectors: vectors, k: k, metric: metric
            )
        }
        return try search(query: query, partition: partition, k: k, nprobe: probe, metric: metric)
    }

    /// Warms the inverted-list cache so the first search does not pay k-means.
    public static func prepare(vectors: VectorBuffer, metric: Metric, nlist: Int? = nil) {
        guard metric != .hamming else { return }
        let count = vectors.count
        guard count > 0, let corpus = vectors.floatPointer.baseAddress else { return }
        let lists = resolvedListCount(requested: nlist ?? defaultListCount, vectorCount: count)
        _ = try? cachedPartition(
            buffer: vectors.buffer,
            corpus: corpus,
            vectorCount: count,
            dim: vectors.dim,
            metric: metric,
            nlist: lists
        )
    }

    public static func invalidateCache(buffer: MTLBuffer) {
        cache.invalidate(bufferID: ObjectIdentifier(buffer))
    }

    public static func invalidate(buffer: MTLBuffer) {
        invalidateCache(buffer: buffer)
    }

    // MARK: - Scan

    private static func scanProbedLists(
        query: UnsafePointer<Float>,
        partition: Partition,
        k: Int,
        nprobe: Int,
        metric: Metric
    ) -> [SearchResult] {
        let nlist = partition.nlist
        let dim = partition.dim

        var queryNormSq: Float = 0
        if metric == .cosine || metric == .l2 {
            vDSP_dotpr(query, 1, query, 1, &queryNormSq, vDSP_Length(dim))
        }

        var centroidDots = [Float](repeating: 0, count: nlist)
        centroidDots.withUnsafeMutableBufferPointer { dots in
            mans_f32_dot_rows(
                partition.centroids,
                query,
                Int64(nlist),
                Int64(dim),
                dots.baseAddress!
            )
        }

        var centroidKeys = [Float](repeating: 0, count: nlist)
        for cluster in 0..<nlist {
            centroidKeys[cluster] = distance(
                dot: centroidDots[cluster],
                queryNormSq: queryNormSq,
                rowNormSq: partition.centroidNormSq[cluster],
                metric: metric
            )
        }

        var probed = smallestIndices(keys: centroidKeys, count: nprobe)
        probed.sort()
        var candidateCount = 0
        for cluster in probed {
            candidateCount += partition.listOffsets[cluster + 1] - partition.listOffsets[cluster]
        }
        guard candidateCount > 0 else { return [] }

        let topK = min(k, candidateCount)
        var heap = TopKHeap(capacity: topK)
        var dots = [Float](repeating: 0, count: candidateCount)
        dots.withUnsafeMutableBufferPointer { dotsBuffer in
            guard let dotsBase = dotsBuffer.baseAddress else { return }
            var cursor = 0
            for cluster in probed {
                let start = partition.listOffsets[cluster]
                let rows = partition.listOffsets[cluster + 1] - start
                guard rows > 0 else { continue }
                mans_f32_dot_rows(
                    partition.packed.advanced(by: start * dim),
                    query,
                    Int64(rows),
                    Int64(dim),
                    dotsBase + cursor
                )
                for local in 0..<rows {
                    let packedIndex = start + local
                    let dist = distance(
                        dot: dotsBase[cursor + local],
                        queryNormSq: queryNormSq,
                        rowNormSq: partition.packedNormSq[packedIndex],
                        metric: metric
                    )
                    heap.append(distance: dist, id: partition.listIDs[packedIndex])
                }
                cursor += rows
            }
        }
        return heap.sortedAscending().map { entry in
            SearchResult(id: "", score: entry.distance, internalID: entry.id)
        }
    }

    @inline(__always)
    private static func distance(
        dot: Float,
        queryNormSq: Float,
        rowNormSq: Float,
        metric: Metric
    ) -> Float {
        switch metric {
        case .innerProduct:
            return -dot
        case .l2:
            return max(0, queryNormSq - 2 * dot + rowNormSq)
        case .cosine:
            let denom = (queryNormSq * rowNormSq).squareRoot()
            return denom < 1e-10 ? 1.0 : 1.0 - dot / denom
        case .hamming:
            return Float.greatestFiniteMagnitude
        }
    }

    /// `count` smallest keys; ties broken by lower index.
    private static func smallestIndices(keys: [Float], count: Int) -> [Int] {
        let limit = min(count, keys.count)
        guard limit > 0 else { return [] }
        var heapKeys = [Float]()
        var heapIDs = [Int]()
        heapKeys.reserveCapacity(limit)
        heapIDs.reserveCapacity(limit)

        func siftDown(from start: Int, size: Int) {
            var position = start
            while true {
                let left = 2 &* position &+ 1
                let right = left &+ 1
                var largest = position
                if left < size,
                    heapKeys[left] > heapKeys[largest]
                        || (heapKeys[left] == heapKeys[largest] && heapIDs[left] > heapIDs[largest])
                {
                    largest = left
                }
                if right < size,
                    heapKeys[right] > heapKeys[largest]
                        || (heapKeys[right] == heapKeys[largest] && heapIDs[right] > heapIDs[largest])
                {
                    largest = right
                }
                if largest == position { return }
                heapKeys.swapAt(position, largest)
                heapIDs.swapAt(position, largest)
                position = largest
            }
        }

        for index in 0..<keys.count {
            let key = keys[index]
            if heapKeys.count < limit {
                heapKeys.append(key)
                heapIDs.append(index)
                var position = heapKeys.count - 1
                while position > 0 {
                    let parent = (position - 1) / 2
                    if heapKeys[position] > heapKeys[parent]
                        || (heapKeys[position] == heapKeys[parent] && heapIDs[position] > heapIDs[parent])
                    {
                        heapKeys.swapAt(position, parent)
                        heapIDs.swapAt(position, parent)
                        position = parent
                    } else {
                        break
                    }
                }
            } else if key < heapKeys[0] || (key == heapKeys[0] && index < heapIDs[0]) {
                heapKeys[0] = key
                heapIDs[0] = index
                siftDown(from: 0, size: limit)
            }
        }
        var order = Array(0..<heapKeys.count)
        order.sort { lhs, rhs in
            if heapKeys[lhs] == heapKeys[rhs] { return heapIDs[lhs] < heapIDs[rhs] }
            return heapKeys[lhs] < heapKeys[rhs]
        }
        var result = [Int]()
        result.reserveCapacity(order.count)
        for index in order {
            result.append(heapIDs[index])
        }
        return result
    }

    private struct TopKHeap {
        struct Entry {
            var distance: Float
            var id: UInt32
        }

        private var entries: [Entry]
        private var size = 0

        init(capacity: Int) {
            entries = [Entry](repeating: Entry(distance: 0, id: 0), count: max(capacity, 1))
        }

        @inline(__always) mutating func append(distance: Float, id: UInt32) {
            if size < entries.count {
                entries[size] = Entry(distance: distance, id: id)
                var position = size
                size += 1
                while position > 0 {
                    let parent = (position - 1) / 2
                    if isWorse(entries[position], than: entries[parent]) {
                        entries.swapAt(position, parent)
                        position = parent
                    } else {
                        break
                    }
                }
            } else if isBetter(distance: distance, id: id, than: entries[0]) {
                entries[0] = Entry(distance: distance, id: id)
                siftDown(from: 0)
            }
        }

        private func isWorse(_ lhs: Entry, than rhs: Entry) -> Bool {
            lhs.distance > rhs.distance || (lhs.distance == rhs.distance && lhs.id > rhs.id)
        }

        private func isBetter(distance: Float, id: UInt32, than worst: Entry) -> Bool {
            distance < worst.distance || (distance == worst.distance && id < worst.id)
        }

        private mutating func siftDown(from start: Int) {
            var position = start
            while true {
                let left = 2 &* position &+ 1
                let right = left &+ 1
                var largest = position
                if left < size, isWorse(entries[left], than: entries[largest]) { largest = left }
                if right < size, isWorse(entries[right], than: entries[largest]) { largest = right }
                if largest == position { return }
                entries.swapAt(position, largest)
                position = largest
            }
        }

        func sortedAscending() -> [Entry] {
            Array(entries.prefix(size)).sorted { lhs, rhs in
                lhs.distance == rhs.distance ? lhs.id < rhs.id : lhs.distance < rhs.distance
            }
        }
    }

    // MARK: - Cache

    private struct CacheKey: Hashable {
        let bufferID: ObjectIdentifier
        let bufferLength: Int
        let count: Int
        let dim: Int
        let metric: Metric
        let nlist: Int
    }

    private final class Store: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [CacheKey: Partition] = [:]
        private var order: [CacheKey] = []
        private let capacity = 8

        func get(_ key: CacheKey) -> Partition? {
            lock.lock()
            defer { lock.unlock() }
            return entries[key]
        }

        func store(_ value: Partition, for key: CacheKey) {
            lock.lock()
            defer { lock.unlock() }
            if entries[key] == nil {
                order.append(key)
                while order.count > capacity {
                    let evicted = order.removeFirst()
                    entries[evicted] = nil
                }
            }
            entries[key] = value
        }

        func invalidate(bufferID: ObjectIdentifier) {
            lock.lock()
            defer { lock.unlock() }
            let doomed = order.filter { $0.bufferID == bufferID }
            for key in doomed { entries[key] = nil }
            order.removeAll { $0.bufferID == bufferID }
        }
    }

    private static let cache = Store()

    private static func cachedPartition(
        buffer: MTLBuffer,
        corpus: UnsafePointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric,
        nlist: Int
    ) throws -> Partition {
        let key = CacheKey(
            bufferID: ObjectIdentifier(buffer),
            bufferLength: buffer.length,
            count: vectorCount,
            dim: dim,
            metric: metric,
            nlist: nlist
        )
        if let cached = cache.get(key) { return cached }
        let built = try buildPartition(
            corpus: corpus, count: vectorCount, dim: dim, metric: metric, nlist: nlist, seed: 42
        )
        cache.store(built, for: key)
        return cache.get(key) ?? built
    }

    private static func buildPartition(
        corpus: UnsafePointer<Float>,
        count: Int,
        dim: Int,
        metric: Metric,
        nlist: Int,
        seed: UInt64
    ) throws -> Partition {
        guard count > 0, dim > 0 else {
            throw ANNSError.constructionFailed("IVF-flat build requires count > 0 and dim > 0")
        }
        let lists = min(max(1, nlist), count)
        let sampleCount = min(count, max(lists, min(4096, max(lists * 16, lists))))
        var sample: [[Float]] = []
        sample.reserveCapacity(sampleCount)
        let stride = max(1, count / sampleCount)
        var cursor = 0
        while sample.count < sampleCount && cursor < count {
            let row = corpus.advanced(by: cursor * dim)
            sample.append(Array(UnsafeBufferPointer(start: row, count: dim)))
            cursor += stride
        }
        while sample.count < sampleCount {
            let rowIndex = sample.count % count
            let row = corpus.advanced(by: rowIndex * dim)
            sample.append(Array(UnsafeBufferPointer(start: row, count: dim)))
        }

        let clustered = try KMeans.cluster(
            vectors: sample,
            k: lists,
            maxIterations: 8,
            metric: metric,
            seed: seed
        )

        let centroids = UnsafeMutablePointer<Float>.allocate(capacity: lists * dim)
        centroids.initialize(repeating: 0, count: lists * dim)
        let centroidNormSq = UnsafeMutablePointer<Float>.allocate(capacity: lists)
        centroidNormSq.initialize(repeating: 0, count: lists)
        for cluster in 0..<lists {
            let src = clustered.centroids[min(cluster, clustered.centroids.count - 1)]
            let dest = centroids.advanced(by: cluster * dim)
            let copyCount = min(dim, src.count)
            src.withUnsafeBufferPointer { buffer in
                dest.update(from: buffer.baseAddress!, count: copyCount)
            }
            var norm: Float = 0
            vDSP_dotpr(dest, 1, dest, 1, &norm, vDSP_Length(dim))
            centroidNormSq[cluster] = norm
        }

        let rowNormSq = UnsafeMutablePointer<Float>.allocate(capacity: count)
        rowNormSq.initialize(repeating: 0, count: count)
        for row in 0..<count {
            var norm: Float = 0
            let rowBase = corpus.advanced(by: row * dim)
            vDSP_dotpr(rowBase, 1, rowBase, 1, &norm, vDSP_Length(dim))
            rowNormSq[row] = norm
        }

        let assignments = assignRows(
            corpus: corpus,
            count: count,
            dim: dim,
            nlist: lists,
            centroids: centroids,
            centroidNormSq: centroidNormSq,
            rowNormSq: rowNormSq,
            metric: metric
        )

        var listSizes = [Int](repeating: 0, count: lists)
        for cluster in assignments { listSizes[cluster] += 1 }
        var listOffsets = [Int](repeating: 0, count: lists + 1)
        for cluster in 0..<lists {
            listOffsets[cluster + 1] = listOffsets[cluster] + listSizes[cluster]
        }
        let listIDs = UnsafeMutablePointer<UInt32>.allocate(capacity: count)
        listIDs.initialize(repeating: 0, count: count)
        var writeAt = listOffsets
        for row in 0..<count {
            let cluster = assignments[row]
            listIDs[writeAt[cluster]] = UInt32(row)
            writeAt[cluster] += 1
        }

        let packed = UnsafeMutablePointer<Float>.allocate(capacity: count * dim)
        packed.initialize(repeating: 0, count: count * dim)
        let packedNormSq = UnsafeMutablePointer<Float>.allocate(capacity: count)
        packedNormSq.initialize(repeating: 0, count: count)
        for packedIndex in 0..<count {
            let sourceRow = Int(listIDs[packedIndex])
            packed.advanced(by: packedIndex * dim).update(
                from: corpus.advanced(by: sourceRow * dim),
                count: dim
            )
            packedNormSq[packedIndex] = rowNormSq[sourceRow]
        }

        return Partition(
            nlist: lists,
            count: count,
            dim: dim,
            metric: metric,
            packed: packed,
            packedNormSq: packedNormSq,
            centroids: centroids,
            centroidNormSq: centroidNormSq,
            rowNormSq: rowNormSq,
            listOffsets: listOffsets,
            listIDs: listIDs,
            listIDCount: count
        )
    }

    private static func assignRows(
        corpus: UnsafePointer<Float>,
        count: Int,
        dim: Int,
        nlist: Int,
        centroids: UnsafePointer<Float>,
        centroidNormSq: UnsafePointer<Float>,
        rowNormSq: UnsafePointer<Float>,
        metric: Metric
    ) -> [Int] {
        let scoreCount = count * nlist
        let scores = UnsafeMutablePointer<Float>.allocate(capacity: scoreCount)
        scores.initialize(repeating: 0, count: scoreCount)
        defer {
            scores.deinitialize(count: scoreCount)
            scores.deallocate()
        }

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            Int32(count),
            Int32(nlist),
            Int32(dim),
            1.0,
            corpus,
            Int32(dim),
            centroids,
            Int32(dim),
            0.0,
            scores,
            Int32(nlist)
        )

        let assignments = UnsafeMutablePointer<Int>.allocate(capacity: count)
        assignments.initialize(repeating: 0, count: count)
        defer {
            assignments.deinitialize(count: count)
            assignments.deallocate()
        }

        let slices = max(1, min(ProcessInfo.processInfo.activeProcessorCount, max(1, count / 512), 32))
        let chunk = (count + slices - 1) / slices
        DispatchQueue.concurrentPerform(iterations: slices) { slice in
            let start = slice * chunk
            let end = min(start + chunk, count)
            guard start < end else { return }
            for row in start..<end {
                var best = 0
                var bestKey = Float.greatestFiniteMagnitude
                let rowScores = scores.advanced(by: row * nlist)
                let qn = rowNormSq[row]
                for cluster in 0..<nlist {
                    let key = distance(
                        dot: rowScores[cluster],
                        queryNormSq: qn,
                        rowNormSq: centroidNormSq[cluster],
                        metric: metric
                    )
                    if key < bestKey || (key == bestKey && cluster < best) {
                        bestKey = key
                        best = cluster
                    }
                }
                assignments[row] = best
            }
        }

        return Array(UnsafeBufferPointer(start: assignments, count: count))
    }
}
