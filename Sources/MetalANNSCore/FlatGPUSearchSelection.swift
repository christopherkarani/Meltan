import Foundation
import Metal

/// Top-K selection machinery for the flat exact-search tiers.
///
/// Split out of FlatGPUSearch.swift to keep both files reviewable;
/// members are internal so the host tier and GPU chunk path can call them.
extension FlatGPUSearch {
    /// Bounded max-heap over (distance, id) pairs; append() is O(1) amortized
    /// once full, sortedAscending() extracts entries in ascending order.
    struct TopKSelector {
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
    /// Bounded max-heap top-K over each query's distance row.
    /// Batched queries select in parallel (rows are independent); a single
    /// large scan is split into per-core chunks whose bounded heaps are then
    /// merged deterministically by (distance, id).
    /// Cost ≈ vectorCount comparisons + O(K·ln(N/K)) replacements.
    static func selectTopK(
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

    struct CandidateEntry {
        var distance: Float
        var id: UInt32
    }

    /// Bounded max-heap top-K over `count` consecutive floats at `row`.
    /// Candidate ids are `idBase + index`; chunked callers must pass their
    /// chunk's start offset so ids stay global row indices.
    /// Returns entries sorted ascending by (distance, id).
    static func boundedTopK(
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

    static func selectRow(
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
    static func selectRowParallel(
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
struct ConcurrentReadPointer<Element>: @unchecked Sendable {
    let pointer: UnsafePointer<Element>

    func advanced(by offset: Int) -> UnsafePointer<Element> {
        pointer.advanced(by: offset)
    }
}

/// Unchecked-Sendable view over a uniquely-owned buffer so concurrent
/// lanes can write distinct indices without tripping COW data races.
struct ConcurrentSlotBuffer<Element>: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<Element>

    subscript(_ index: Int) -> Element {
        get { buffer[index] }
        nonmutating set { buffer[index] = newValue }
    }
}
