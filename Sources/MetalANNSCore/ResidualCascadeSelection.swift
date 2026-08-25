import Accelerate
import Foundation

/// Parallel/serial selection machinery for the exact cascade.
///
/// Split out of ResidualCascade.swift for file-size constraints; members
/// are internal so the search entry point can call them.
extension ResidualCascade {
    struct ResolveInput {
        let lowerBounds: [Float]
        let corpus: UnsafePointer<Float>
        let dimensionCount: Int
        let query: [Float]
        let queryNormSq: Float
        let aux: BoundBuffer
        let metric: Metric
        let effectiveK: Int
        let rowCount: Int
        let recordPhase: (String, DispatchTime) -> Void
        let statsEnabled: Bool
    }

    // MARK: - Selection helpers

    /// Indices of the `take` smallest values, ascending by value (ties by index).
    /// Parallel: per-chunk bounded heaps via concurrentPerform, then merge.
    static func selectSmallestIndices(lowerBounds: [Float], take: Int) -> [UInt32] {
        let totalRows = lowerBounds.count
        let capacity = min(take, totalRows)
        guard capacity > 0 else { return [] }

        if totalRows <= 64_000 {
            return serialSmallestIndices(lowerBounds: lowerBounds, take: capacity)
        }

        let workers = ProcessInfo.processInfo.activeProcessorCount
        let lanes = max(1, min(workers, totalRows / 32_000))
        if lanes <= 1 {
            return serialSmallestIndices(lowerBounds: lowerBounds, take: capacity)
        }
        let chunkSize = (totalRows + lanes - 1) / lanes
        let laneValuesBuffer = UnsafeMutableBufferPointer<[Float]>.allocate(capacity: lanes)
        let laneIDsBuffer = UnsafeMutableBufferPointer<[UInt32]>.allocate(capacity: lanes)
        defer {
            laneValuesBuffer.deallocate()
            laneIDsBuffer.deallocate()
        }
        for lane in 0..<lanes {
            laneValuesBuffer[lane] = []
            laneIDsBuffer[lane] = []
        }

        lowerBounds.withUnsafeBufferPointer { valuesBuf in
            DispatchQueue.concurrentPerform(iterations: lanes) { lane in
                let start = lane * chunkSize
                let end = min(totalRows, start + chunkSize)
                guard start < end else { return }
                var heapValues = ContiguousArray<Float>()
                var heapIndices = ContiguousArray<UInt32>()
                heapValues.reserveCapacity(capacity)
                heapIndices.reserveCapacity(capacity)

                @inline(__always) func siftDownFrom(_ position: Int) {
                    var position = position
                    while true {
                        let leftChild = 2 * position + 1
                        let rightChild = leftChild + 1
                        var largest = position
                        if leftChild < heapValues.count, heapValues[leftChild] > heapValues[largest] { largest = leftChild }
                        if rightChild < heapValues.count, heapValues[rightChild] > heapValues[largest] { largest = rightChild }
                        if largest == position { return }
                        heapValues.swapAt(position, largest)
                        heapIndices.swapAt(position, largest)
                        position = largest
                    }
                }

                for index in start..<end {
                    let value = valuesBuf[index]
                    if heapValues.count < capacity {
                        heapValues.append(value)
                        heapIndices.append(UInt32(index))
                        var position = heapValues.count - 1
                        while position > 0 {
                            let parent = (position - 1) / 2
                            if heapValues[position] > heapValues[parent] {
                                heapValues.swapAt(position, parent)
                                heapIndices.swapAt(position, parent)
                                position = parent
                            } else { break }
                        }
                    } else if value < heapValues[0] {
                        heapValues[0] = value
                        heapIndices[0] = UInt32(index)
                        siftDownFrom(0)
                    }
                }
                laneValuesBuffer[lane] = Array(heapValues)
                laneIDsBuffer[lane] = Array(heapIndices)
            }
        }

        var pairs = [(value: Float, index: UInt32)]()
        pairs.reserveCapacity(lanes * capacity)
        for lane in 0..<lanes {
            for (offset, value) in laneValuesBuffer[lane].enumerated() {
                pairs.append((value, laneIDsBuffer[lane][offset]))
            }
        }
        pairs.sort { $0.value == $1.value ? $0.index < $1.index : $0.value < $1.value }
        return pairs.prefix(capacity).map { $0.index }
    }

    /// Serial bounded-heap variant (small inputs only).
    static func serialSmallestIndices(lowerBounds: [Float], take: Int) -> [UInt32] {
        let totalRows = lowerBounds.count
        var heapValues = ContiguousArray<Float>()
        var heapIndices = ContiguousArray<UInt32>()
        heapValues.reserveCapacity(take)
        heapIndices.reserveCapacity(take)

        @inline(__always) func siftDownFrom(_ position: Int) {
            var position = position
            while true {
                let leftChild = 2 * position + 1
                let rightChild = leftChild + 1
                var largest = position
                if leftChild < heapValues.count, heapValues[leftChild] > heapValues[largest] { largest = leftChild }
                if rightChild < heapValues.count, heapValues[rightChild] > heapValues[largest] { largest = rightChild }
                if largest == position { return }
                heapValues.swapAt(position, largest)
                heapIndices.swapAt(position, largest)
                position = largest
            }
        }

        for index in 0..<totalRows {
            let value = lowerBounds[index]
            if heapValues.count < take {
                heapValues.append(value)
                heapIndices.append(UInt32(index))
                var position = heapValues.count - 1
                while position > 0 {
                    let parent = (position - 1) / 2
                    if heapValues[position] > heapValues[parent] {
                        heapValues.swapAt(position, parent)
                        heapIndices.swapAt(position, parent)
                        position = parent
                    } else { break }
                }
            } else if value < heapValues[0] {
                heapValues[0] = value
                heapIndices[0] = UInt32(index)
                siftDownFrom(0)
            }
        }
        var pairs = zip(heapValues, heapIndices).map { (value: $0.0, index: $0.1) }
        pairs.sort { $0.value == $1.value ? $0.index < $1.index : $0.value < $1.value }
        return pairs.map { $0.index }
    }

    /// Ascending top-k with deterministic tie-break by id. Uses a bounded
    /// max-heap so cost is O(n·log k) rather than a full sort of every
    /// rescored candidate.
    static func topKResults(
        of scored: [(distance: Float, id: UInt32)], take: Int
    ) -> [SearchResult] {
        let selected = min(take, scored.count)
        guard selected > 0 else { return [] }

        var heapDistances = ContiguousArray<Float>()
        var heapIDs = ContiguousArray<UInt32>()
        heapDistances.reserveCapacity(selected)
        heapIDs.reserveCapacity(selected)

        @inline(__always) func siftDownFrom(_ position: Int, size: Int) {
            var position = position
            while true {
                let leftChild = 2 * position + 1
                let rightChild = leftChild + 1
                var largest = position
                if leftChild < size,
                    heapDistances[leftChild] > heapDistances[largest]
                { largest = leftChild }
                if rightChild < size,
                    heapDistances[rightChild] > heapDistances[largest]
                { largest = rightChild }
                if largest == position { return }
                heapDistances.swapAt(position, largest)
                heapIDs.swapAt(position, largest)
                position = largest
            }
        }

        for entry in scored {
            if heapDistances.count < selected {
                heapDistances.append(entry.distance)
                heapIDs.append(entry.id)
                var position = heapDistances.count - 1
                while position > 0 {
                    let parent = (position - 1) / 2
                    if heapDistances[position] > heapDistances[parent] {
                        heapDistances.swapAt(position, parent)
                        heapIDs.swapAt(position, parent)
                        position = parent
                    } else { break }
                }
            } else if entry.distance < heapDistances[0] {
                heapDistances[0] = entry.distance
                heapIDs[0] = entry.id
                siftDownFrom(0, size: heapDistances.count)
            }
        }

        var extracted = [(distance: Float, id: UInt32)]()
        extracted.reserveCapacity(selected)
        var size = selected
        while size > 0 {
            extracted.append((heapDistances[0], heapIDs[0]))
            size -= 1
            if size > 0 {
                heapDistances[0] = heapDistances[size]
                heapIDs[0] = heapIDs[size]
                siftDownFrom(0, size: size)
            }
        }
        extracted.reverse()

        var hasTies = false
        if extracted.count > 1 {
            for index in 1..<extracted.count where extracted[index].distance == extracted[index - 1].distance {
                hasTies = true
                break
            }
        }
        if hasTies {
            extracted.sort {
                $0.distance == $1.distance ? $0.id < $1.id : $0.distance < $1.distance
            }
        }
        return extracted.map { entry in
            SearchResult(id: "", score: entry.distance, internalID: entry.id)
        }
    }

    /// Rows whose bound beats the cutoff and that were not already rescored
    /// as seeds. Parallel over workers with per-lane buffers merged in order,
    /// so the output id order is deterministic.
    static func collectSurvivors(
        lowerBounds: [Float],
        seedMark: [Bool],
        cutoff: Float,
        rowCount: Int
    ) -> [UInt32] {
        let workers = ProcessInfo.processInfo.activeProcessorCount
        let lanes = max(1, min(workers, rowCount / 64_000))
        guard lanes > 1 else {
            var survivors: [UInt32] = []
            survivors.reserveCapacity(rowCount / 8)
            for rowIndex in 0..<rowCount where !seedMark[rowIndex] && lowerBounds[rowIndex] < cutoff {
                survivors.append(UInt32(rowIndex))
            }
            return survivors
        }

        let chunkSize = (rowCount + lanes - 1) / lanes
        let laneResultsBuffer = UnsafeMutableBufferPointer<[UInt32]>.allocate(capacity: lanes)
        defer { laneResultsBuffer.deallocate() }
        for lane in 0..<lanes {
            laneResultsBuffer[lane] = []
        }
        lowerBounds.withUnsafeBufferPointer { boundsBuf in
            seedMark.withUnsafeBufferPointer { markBuf in
                DispatchQueue.concurrentPerform(iterations: lanes) { lane in
                    let start = lane * chunkSize
                    let end = min(rowCount, start + chunkSize)
                    guard start < end else { return }
                    var local: [UInt32] = []
                    local.reserveCapacity((end - start) / 8 + 16)
                    for rowIndex in start..<end where !markBuf[rowIndex] && boundsBuf[rowIndex] < cutoff {
                        local.append(UInt32(rowIndex))
                    }
                    laneResultsBuffer[lane] = local
                }
            }
        }
        var survivors: [UInt32] = []
        survivors.reserveCapacity(rowCount / 8)
        for lane in 0..<lanes {
            survivors.append(contentsOf: laneResultsBuffer[lane])
        }
        return survivors
    }

    /// Rescores seeds, prunes by provable bound, rescores survivors, and
    /// returns the exact top-k. Hosts the optional diagnostics passes.
    static func resolveTopK(_ input: ResolveInput) -> [SearchResult]? {
        let lowerBounds = input.lowerBounds
        let corpus = input.corpus
        let dimensionCount = input.dimensionCount
        let query = input.query
        let queryNormSq = input.queryNormSq
        let aux = input.aux
        let metric = input.metric
        let effectiveK = input.effectiveK
        let rowCount = input.rowCount
        let recordPhase = input.recordPhase
        let statsEnabled = input.statsEnabled

        // ---- Phases 2+3: seeds, exact rescore, cutoff ----------------------
        let phase23Start = DispatchTime.now()
        let seedTarget = min(max(4 * effectiveK, 256), rowCount)
        let seedIDs = selectSmallestIndices(lowerBounds: lowerBounds, take: seedTarget)
        var scored: [(distance: Float, id: UInt32)] = []
        scored.reserveCapacity(seedTarget * 4)
        exactRescore(
            ids: seedIDs, corpus: corpus, dimensionCount: dimensionCount,
            query: query, queryNormSq: queryNormSq, metric: metric, into: &scored
        )
        scored.sort { $0.distance < $1.distance }
        let cutoffIndex = min(effectiveK, scored.count) - 1
        guard cutoffIndex >= 0 else { return [] }
        let cutoff = scored[cutoffIndex].distance
        recordPhase("p23_seeds_and_cutoff", phase23Start)

        // ---- Phase 4: survivors -------------------------------------------
        let phase4Start = DispatchTime.now()
        var seedMark = [Bool](repeating: false, count: rowCount)
        for id in seedIDs { seedMark[Int(id)] = true }
        let survivors = collectSurvivors(
            lowerBounds: lowerBounds, seedMark: seedMark,
            cutoff: cutoff, rowCount: rowCount
        )

        if statsEnabled {
            FileHandle.standardError.write(Data(
                "[ResidualCascade] rows=\(rowCount) seeds=\(seedIDs.count) cutoff=\(cutoff) survivors=\(survivors.count) tailAvg=\(aux.avgTailNorm) rowNormAvg=\(aux.avgRowNorm)\n".utf8
            ))
        }
        recordPhase("p4_survivor_scan", phase4Start)

        // ---- Phase 5: exact survivor rescore -------------------------------
        let phase5Start = DispatchTime.now()
        exactRescore(
            ids: survivors, corpus: corpus, dimensionCount: dimensionCount,
            query: query, queryNormSq: queryNormSq, metric: metric, into: &scored
        )
        recordPhase("p5_survivor_rescore", phase5Start)

        // ---- Phase 6: top-k -------------------------------------------------
        let phase6Start = DispatchTime.now()
        let results = topKResults(of: scored, take: effectiveK)

        if ProcessInfo.processInfo.environment["METALANNS_VERIFY_BOUNDS"] == "1" {
            // Diagnostic: brute-force verify bound soundness on every row.
            let verifyWorkers = ProcessInfo.processInfo.activeProcessorCount
            var maxViolation: Float = 0
            var violationCount = 0
            let trueDistances = UnsafeMutableBufferPointer<Float>.allocate(capacity: rowCount)
            defer { trueDistances.deallocate() }
            let capturedQueryNormSq = queryNormSq
            DispatchQueue.concurrentPerform(iterations: verifyWorkers) { worker in
                let start = (rowCount * worker) / verifyWorkers
                let end = (rowCount * (worker + 1)) / verifyWorkers
                guard start < end else { return }
                query.withUnsafeBufferPointer { qBuf in
                    for rowIndex in start..<end {
                        let rowBase = corpus + rowIndex * dimensionCount
                        var dotQV: Float = 0
                        var nSq: Float = 0
                        for offset in 0..<dimensionCount {
                            dotQV += qBuf[offset] * rowBase[offset]
                            nSq += rowBase[offset] * rowBase[offset]
                        }
                        trueDistances[rowIndex] = finalizeScore(
                            dotQV: dotQV, normVSq: nSq,
                            metric: metric, queryNormSq: capturedQueryNormSq
                        )
                    }
                }
            }
            for rowIndex in 0..<rowCount {
                if lowerBounds[rowIndex] > trueDistances[rowIndex] {
                    violationCount += 1
                    maxViolation = max(maxViolation, lowerBounds[rowIndex] - trueDistances[rowIndex])
                }
            }
            let verifyLine = "[ResidualCascade] VERIFY: violations=\(violationCount)"
                + " max=\(maxViolation) dim=\(dimensionCount) w=\(aux.headWidth)"
                + " lb0=\(lowerBounds[0]) td0=\(trueDistances[0])\n"
            FileHandle.standardError.write(Data(verifyLine.utf8))
        }
        recordPhase("p6_topk", phase6Start)

        return results
    }
}
