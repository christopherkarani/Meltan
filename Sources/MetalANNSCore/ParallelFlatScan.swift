import Foundation
import MetalANNSC

/// Exact multithreaded flat scan over a row-major Float32 corpus.
///
/// Splits rows across worker threads, computes exact dots per slice via NEON
/// kernels (`MetalANNSC`), keeps each slice's exact top-K, and merges the
/// sorted per-slice lists. Because each slice contributes its own true top-K,
/// the merged result is the global true top-K: recall is identical to a
/// single-threaded brute-force pass.
enum ParallelFlatScan {
    /// Minimum rows worth handing to an extra thread.
    static let minRowsPerSlice = 2048
    static let maxSlices = 32

    struct Entry {
        var distance: Float
        var id: UInt32
    }

    /// Scoped work descriptor for the parallel join.
    ///
    /// `@unchecked Sendable` is sound here: raw pointers point at caller-owned
    /// memory that stays alive for the duration of the concurrentPerform join,
    /// each worker writes only its own scratch/result slots, and no field
    /// mutates after construction.
    private final class SliceWork: @unchecked Sendable {
        let query: UnsafePointer<Float>
        let corpus: UnsafePointer<Float>
        let norms: UnsafePointer<Float>?
        let queryNormSq: Float
        let dim: Int
        let topK: Int
        let metric: Metric
        let chunkSize: Int
        let dotsBase: UnsafeMutablePointer<Float>
        let results: UnsafeMutableBufferPointer<[Entry]>

        init(
            query: UnsafePointer<Float>,
            corpus: UnsafePointer<Float>,
            norms: UnsafePointer<Float>?,
            queryNormSq: Float,
            dim: Int,
            topK: Int,
            metric: Metric,
            chunkSize: Int,
            dotsBase: UnsafeMutablePointer<Float>,
            results: UnsafeMutableBufferPointer<[Entry]>
        ) {
            self.query = query
            self.corpus = corpus
            self.norms = norms
            self.queryNormSq = queryNormSq
            self.dim = dim
            self.topK = topK
            self.metric = metric
            self.chunkSize = chunkSize
            self.dotsBase = dotsBase
            self.results = results
        }

        func run(slice: Int, vectorCount: Int) {
            let start = slice * chunkSize
            let end = min(start + chunkSize, vectorCount)
            guard start < end else { return }
            let rows = end - start
            let dots = dotsBase + slice * chunkSize
            mans_f32_dot_rows(corpus + start * dim, query, Int64(rows), Int64(dim), dots)
            results[slice] = selectTopK(
                dots: dots,
                corpusRowOffset: start,
                rowCount: rows,
                dim: dim,
                norms: norms,
                queryNormSq: queryNormSq,
                topK: topK,
                metric: metric
            )
        }
    }

    /// Worker count for a given corpus size.
    static func sliceCount(for vectorCount: Int) -> Int {
        let cores = ProcessInfo.processInfo.activeProcessorCount
        let byWork = max(1, vectorCount / minRowsPerSlice)
        return max(1, min(cores, maxSlices, byWork))
    }

    /// Exact top-`topK` entries (ascending distance).
    ///
    /// - Parameters:
    ///   - corpus: row-major Float32 matrix, `vectorCount` rows × `dim` columns.
    ///   - norms: cached squared row norms; required for `.cosine`/`.l2`.
    ///   - queryNormSq: squared norm of `query`; used by `.cosine`/`.l2`.
    static func search(
        query: UnsafePointer<Float>,
        corpus: UnsafePointer<Float>,
        norms: UnsafePointer<Float>?,
        queryNormSq: Float,
        vectorCount: Int,
        dim: Int,
        topK: Int,
        metric: Metric
    ) -> [Entry] {
        guard vectorCount > 0, topK > 0 else { return [] }
        let k = min(topK, vectorCount)

        let slices = sliceCount(for: vectorCount)
        if slices <= 1 {
            return scanSlice(
                query: query, corpus: corpus, norms: norms, queryNormSq: queryNormSq,
                range: 0..<vectorCount, dim: dim, topK: k, metric: metric
            )
        }

        let chunkSize = (vectorCount + slices - 1) / slices

        // Manual storage so worker closures can write disjoint slots without
        // capturing non-Sendable Swift collections.
        let resultsStorage = UnsafeMutableBufferPointer<[Entry]>.allocate(capacity: slices)
        resultsStorage.initialize(repeating: [])
        defer {
            resultsStorage.deinitialize()
            resultsStorage.deallocate()
        }

        // One contiguous scratch region: slice s owns dots[s*chunkSize...].
        var scratch = [Float](repeating: 0, count: slices * chunkSize)

        scratch.withUnsafeMutableBufferPointer { scratchBuffer in
            let context = SliceWork(
                query: query,
                corpus: corpus,
                norms: norms,
                queryNormSq: queryNormSq,
                dim: dim,
                topK: k,
                metric: metric,
                chunkSize: chunkSize,
                dotsBase: scratchBuffer.baseAddress!,
                results: resultsStorage
            )
            DispatchQueue.concurrentPerform(iterations: slices) { slice in
                context.run(slice: slice, vectorCount: vectorCount)
            }
        }

        var sliceLists = [[Entry]](repeating: [], count: slices)
        for index in 0..<slices { sliceLists[index] = resultsStorage[index] }

        // K-way merge of ascending lists; each list is its slice's exact
        // top-k, so merging and truncating yields the global exact top-k.
        return mergeSorted(sliceLists, limit: k)
    }

    /// Single-threaded scan over one row range (also the fallback path).
    static func scanSlice(
        query: UnsafePointer<Float>,
        corpus: UnsafePointer<Float>,
        norms: UnsafePointer<Float>?,
        queryNormSq: Float,
        range: Range<Int>,
        dim: Int,
        topK: Int,
        metric: Metric
    ) -> [Entry] {
        let rows = range.count
        var dots = [Float](repeating: 0, count: rows)
        dots.withUnsafeMutableBufferPointer { buffer in
            mans_f32_dot_rows(corpus + range.lowerBound * dim, query, Int64(rows), Int64(dim), buffer.baseAddress!)
        }
        return dots.withUnsafeBufferPointer { buffer in
            selectTopK(
                dots: buffer.baseAddress!,
                corpusRowOffset: range.lowerBound,
                rowCount: rows,
                dim: dim,
                norms: norms,
                queryNormSq: queryNormSq,
                topK: topK,
                metric: metric
            )
        }
    }

    /// Builds the slice's exact ascending top-K from raw dot products.
    private static func selectTopK(
        dots: UnsafePointer<Float>,
        corpusRowOffset: Int,
        rowCount: Int,
        dim: Int,
        norms: UnsafePointer<Float>?,
        queryNormSq: Float,
        topK: Int,
        metric: Metric
    ) -> [Entry] {
        var entries = [Entry]()
        entries.reserveCapacity(topK)

        @inline(__always) func siftUp(_ index: Int) {
            var position = index
            while position > 0 {
                let parent = (position - 1) / 2
                if entries[position].distance > entries[parent].distance {
                    entries.swapAt(position, parent)
                    position = parent
                } else {
                    break
                }
            }
        }
        @inline(__always) func siftDown() {
            var position = 0
            let count = entries.count
            while true {
                let left = 2 &* position &+ 1
                let right = left &+ 1
                var largest = position
                if left < count, entries[left].distance > entries[largest].distance { largest = left }
                if right < count, entries[right].distance > entries[largest].distance { largest = right }
                if largest == position { return }
                entries.swapAt(position, largest)
                position = largest
            }
        }
        @inline(__always) func append(distance: Float, id: UInt32) {
            if entries.count < topK {
                entries.append(Entry(distance: distance, id: id))
                siftUp(entries.count - 1)
            } else if distance < entries[0].distance {
                entries[0] = Entry(distance: distance, id: id)
                siftDown()
            }
        }

        switch metric {
        case .innerProduct:
            // Distance is negated dot product; ordering equals descending dot.
            for row in 0..<rowCount {
                append(distance: -dots[row], id: UInt32(corpusRowOffset + row))
            }
        case .l2:
            let normsBase = norms!
            for row in 0..<rowCount {
                let d = max(0, queryNormSq - 2 * dots[row] + normsBase[corpusRowOffset + row])
                append(distance: d, id: UInt32(corpusRowOffset + row))
            }
        case .cosine:
            let normsBase = norms!
            for row in 0..<rowCount {
                // Matches the historical host formula bit-for-bit:
                // denom = sqrt(queryNormSq * rowNormSq).
                let denom = (queryNormSq * normsBase[corpusRowOffset + row]).squareRoot()
                let d = denom < 1e-10 ? 1.0 : 1.0 - dots[row] / denom
                append(distance: d, id: UInt32(corpusRowOffset + row))
            }
        case .hamming:
            break
        }

        return heapSortedAscending(entries)
    }

    /// Extracts heap entries in ascending distance order.
    static func heapSortedAscending(_ heap: [Entry]) -> [Entry] {
        var entries = heap
        var count = entries.count
        while count > 1 {
            entries.swapAt(0, count - 1)
            count -= 1
            var position = 0
            while true {
                let left = 2 &* position &+ 1
                let right = left &+ 1
                var largest = position
                if left < count, entries[left].distance > entries[largest].distance { largest = left }
                if right < count, entries[right].distance > entries[largest].distance { largest = right }
                if largest == position { break }
                entries.swapAt(position, largest)
                position = largest
            }
        }
        return entries
    }

    /// Merges already-ascending entry lists into one ascending list capped at `limit`.
    static func mergeSorted(_ lists: [[Entry]], limit: Int) -> [Entry] {
        let liveLists = lists.filter { !$0.isEmpty }
        guard !liveLists.isEmpty else { return [] }
        guard liveLists.count > 1 else {
            return Array(liveLists[0].prefix(limit))
        }

        var cursors = [Int](repeating: 0, count: liveLists.count)
        var merged = [Entry]()
        merged.reserveCapacity(limit)

        while merged.count < limit {
            var bestList = -1
            var bestDistance = Float.infinity
            for (index, list) in liveLists.enumerated() where cursors[index] < list.count {
                let candidate = list[cursors[index]].distance
                if candidate < bestDistance {
                    bestDistance = candidate
                    bestList = index
                }
            }
            guard bestList >= 0 else { break }
            merged.append(liveLists[bestList][cursors[bestList]])
            cursors[bestList] += 1
        }
        return merged
    }
}
