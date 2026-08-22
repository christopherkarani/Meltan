import Accelerate
import Foundation
import MetalANNSC
import simd

/// Exact nearest-neighbor scan accelerated by an int8 quantized prefilter.
///
/// The prefilter computes *exact integer* dot products between int8 codes and
/// an int8-quantized query, then derives provable lower bounds on each true
/// distance from quantization error arithmetic:
///
///   q_d = qs·(q̂_d + f_qd), |f_qd| ≤ 1/2        (query scale qs)
///   v_d = u_r·(ĉ_rd + f_vd), |f_vd| ≤ 1/2      (row scale u_r)
///   dot(q,v) ∈ qs·u_r·D_r ± qs·(w_r + u_r·|q̂|₁/2)
///   D_r = Σ q̂_d·ĉ_rd  (exact int32 accumulation, NEON kernel)
///
/// Each parallel slice computes its rows' exact dots and immediately ranks
/// them by distance lower bound, keeping a bounded candidate heap. The merged
/// candidates are rescored in full fp32. When the budget-th smallest lower
/// bound is at least the k-th best rescored distance, every excluded row
/// provably cannot belong to the true top-k and the result IS exact.
/// Otherwise the budget grows ×4 (dots are cached per slice) and selection
/// repeats. Once the budget covers all rows, the rescore is itself a full
/// brute-force pass. Every path returns brute-force-exact results.
enum BoundedExactScan {
    /// Minimum corpus size where the int8 prefilter beats scanning fp32 directly.
    static let minVectorCount = 16384

    /// Returns nil when the caller should fall back to the plain fp32 scan.
    static func search(
        query: UnsafePointer<Float>,
        corpus: UnsafePointer<Float>,
        codes: Int8CodeBuffer,
        norms: UnsafePointer<Float>?,
        queryNormSq: Float,
        vectorCount: Int,
        dim: Int,
        topK: Int,
        metric: Metric
    ) -> [ParallelFlatScan.Entry]? {
        guard vectorCount >= minVectorCount, dim > 0, topK > 0 else { return nil }
        guard metric != .hamming else { return nil }
        switch metric {
        case .cosine, .l2:
            guard norms != nil else { return nil }
        case .innerProduct, .hamming:
            break
        }

        let k = min(topK, vectorCount)

        // ---- Query quantization ------------------------------------------
        var queryMaxAbs: Float = 0
        vDSP_maxmgv(query, 1, &queryMaxAbs, vDSP_Length(dim))
        let queryScale = queryMaxAbs > 0 ? Double(queryMaxAbs) / 127.0 : 1.0
        var queryCodes = [Int8](repeating: 0, count: dim)
        var queryAbsSum: Int64 = 0
        for d in 0..<dim {
            let scaled = Double(query[d]) / queryScale
            var rounded = Int32(scaled >= 0 ? scaled + 0.5 : scaled - 0.5)
            if rounded > 127 { rounded = 127 } else if rounded < -127 { rounded = -127 }
            queryCodes[d] = Int8(rounded)
            queryAbsSum += Int64(rounded < 0 ? -rounded : rounded)
        }

        // Query-side coefficients. Keys ARE distance lower bounds in true
        // units for every metric:
        //   dotHi(r) = u_r·(qsDot·D + qsQabsHalfU') + qsW·w_r
        // with qsQabsHalfU' = qs·|q̂|₁/2 folded per metric below.
        let qsF = Float(queryScale)
        let qabsHalfF = Float(queryAbsSum) / 2

        let context = ScanContext(
            queryCodes: queryCodes,
            codes: codes.codesBase,
            uPlane: codes.uBase,
            wPlane: codes.wBase,
            normSqPlane: codes.normSqBase,
            invSqrtPlane: codes.invSqrtBase,
            dim: dim,
            metric: metric,
            slices: ParallelFlatScan.sliceCount(for: vectorCount),
            chunkSize: (vectorCount + ParallelFlatScan.sliceCount(for: vectorCount) - 1)
                / ParallelFlatScan.sliceCount(for: vectorCount),
            qsDot: qsF,
            qsW: qsF,
            qsQabsHalf: qsF * qabsHalfF,
            queryNormSq: queryNormSq,
            cosFDot: qsF * (queryNormSq > 1e-20 ? 1.0 / sqrt(queryNormSq) : 0),
            cosFH: qsF * qabsHalfF * (queryNormSq > 1e-20 ? 1.0 / sqrt(queryNormSq) : 0),
            cosFW: qsF * (queryNormSq > 1e-20 ? 1.0 / sqrt(queryNormSq) : 0),
            invSqrtQueryNorm: queryNormSq > 1e-20 ? Float(1.0 / sqrt(Double(queryNormSq))) : 0,
            queryNormSqD: Double(queryNormSq)
        )

        // ---- Adaptive candidate rounds ------------------------------------
        var budget = max(256, k * 16)
        while true {
            let selection = selectCandidates(context: context, vectorCount: vectorCount, budget: budget)
            let rescored = rescore(
                ids: selection.ids, corpus: corpus, query: query, norms: norms,
                queryNormSq: queryNormSq, dim: dim, metric: metric
            )

            if rescored.count < k {
                return rescored  // already sorted ascending
            }

            let tau = rescored[k - 1].distance
            // Margin absorbs float rounding of tau and key arithmetic.
            let margin = max(1e-7, abs(Double(tau)) * 1e-6)
            let proven =
                selection.cutoff >= Double(tau) + margin
                || selection.ids.count >= vectorCount
            if proven {
                return Array(rescored.prefix(k))
            }
            if budget >= vectorCount { break }  // unreachable; defensive
            budget = min(budget * 4, vectorCount)
        }
        return nil
    }

    // MARK: - Fused dots + selection

    private struct SelectionResult {
        let ids: [UInt32]
        /// Lower-bound value of the last admitted row (budget-th smallest).
        let cutoff: Double
    }

    private typealias HeapEntry = (key: Float, id: UInt32)

    /// One parallel pass: each slice computes exact int8 dots for its rows and
    /// immediately keeps the `perSliceBudget` rows with smallest lower bounds.
    private static func selectCandidates(
        context: ScanContext,
        vectorCount: Int,
        budget: Int
    ) -> SelectionResult {
        let slices = context.slices
        let chunkSize = context.chunkSize
        let perSliceBudget = min(budget, chunkSize)
        var sliceResults = [[HeapEntry]](repeating: [], count: slices)

        let dotsScratch = UnsafeMutablePointer<Float>.allocate(capacity: slices * chunkSize)
        defer { dotsScratch.deallocate() }

        DispatchQueue.concurrentPerform(iterations: slices) { slice in
            let start = slice * chunkSize
            let end = min(start + chunkSize, vectorCount)
            guard start < end else { return }
            let rows = end - start
            let dots = dotsScratch + slice * chunkSize

            context.queryCodes.withUnsafeBufferPointer { queryBuffer in
                mans_i8_dot_rows_f32(
                    context.codes + start * context.dim,
                    queryBuffer.baseAddress!,
                    Int64(rows),
                    Int64(context.dim),
                    dots
                )
            }

            var heap = [HeapEntry]()
            heap.reserveCapacity(perSliceBudget)
            let capacity = perSliceBudget

            switch context.metric {
            case .innerProduct:
                // dLo = -(qsDot·u·D + qsQabsHalf·u + qsW·w)
                let uP = context.uPlane + start
                let wP = context.wPlane + start
                for offset in 0..<rows {
                    let u = uP[offset]
                    let key =
                        -(context.qsDot * u * dots[offset]
                        + context.qsQabsHalf * u
                        + context.qsW * wP[offset])
                    admit(key: key, id: UInt32(start + offset), heap: &heap, capacity: capacity)
                }
            case .l2:
                // dLo = qn + vn - 2·qsDot·u·D - 2·qsQabsHalf·u - 2·qsW·w
                let uP = context.uPlane + start
                let wP = context.wPlane + start
                let nP = context.normSqPlane + start
                let qn = context.queryNormSqD
                let twoQsDot = 2 * context.qsDot
                let twoQsqh = 2 * context.qsQabsHalf
                let twoQsW = 2 * context.qsW
                for offset in 0..<rows {
                    let u = uP[offset]
                    let key =
                        qn + Double(nP[offset])
                        - Double(twoQsDot * u * dots[offset])
                        - Double(twoQsqh * u)
                        - Double(twoQsW * wP[offset])
                    admit(key: Float(key), id: UInt32(start + offset), heap: &heap, capacity: capacity)
                }
            case .cosine:
                // dLo = 1 - (cosFDot·u·D + cosFH·u + cosFW·w) · invSqrtRow
                let uP = context.uPlane + start
                let wP = context.wPlane + start
                let iP = context.invSqrtPlane + start
                for offset in 0..<rows {
                    let inner =
                        context.cosFDot * uP[offset] * dots[offset]
                        + context.cosFH * uP[offset]
                        + context.cosFW * wP[offset]
                    admit(
                        key: 1.0 - inner * iP[offset],
                        id: UInt32(start + offset),
                        heap: &heap,
                        capacity: capacity
                    )
                }
            case .hamming:
                break
            }

            sliceResults[slice] = heapSortedAscending(heap)
        }

        return mergeAscending(sliceResults, budget: budget)
    }

    @inline(__always)
    private static func admit(
        key: Float, id: UInt32, heap: inout [HeapEntry], capacity: Int
    ) {
        if heap.count < capacity {
            heap.append((key, id))
            siftUpHeap(&heap, heap.count - 1)
        } else if key < heap[0].key {
            heap[0] = (key, id)
            siftDownHeap(&heap, 0, heap.count)
        }
    }

    private static func siftUpHeap(_ heap: inout [HeapEntry], _ index: Int) {
        var position = index
        while position > 0 {
            let parent = (position - 1) / 2
            if heap[position].key > heap[parent].key {
                heap.swapAt(position, parent)
                position = parent
            } else {
                break
            }
        }
    }

    private static func siftDownHeap(_ heap: inout [HeapEntry], _ start: Int, _ size: Int) {
        var position = start
        while true {
            let left = 2 &* position &+ 1
            let right = left &+ 1
            var largest = position
            if left < size, heap[left].key > heap[largest].key { largest = left }
            if right < size, heap[right].key > heap[largest].key { largest = right }
            if largest == position { return }
            heap.swapAt(position, largest)
            position = largest
        }
    }

    /// Heapsort extraction of a max-heap into ascending order.
    private static func heapSortedAscending(_ heap: [HeapEntry]) -> [HeapEntry] {
        var entries = heap
        var count = entries.count
        while count > 1 {
            entries.swapAt(0, count - 1)
            count -= 1
            siftDownHeap(&entries, 0, count)
        }
        return entries
    }

    private static func mergeAscending(
        _ lists: [[HeapEntry]], budget: Int
    ) -> SelectionResult {
        var cursors = [Int](repeating: 0, count: lists.count)
        var mergedIds = [UInt32]()
        mergedIds.reserveCapacity(budget)
        var cutoff = Double.infinity
        var remaining = budget

        while remaining > 0 {
            var bestList = -1
            var bestKey = Float.infinity
            for index in 0..<lists.count where cursors[index] < lists[index].count {
                let candidate = lists[index][cursors[index]].key
                if candidate < bestKey {
                    bestKey = candidate
                    bestList = index
                }
            }
            guard bestList >= 0 else { break }
            mergedIds.append(lists[bestList][cursors[bestList]].id)
            cutoff = Double(bestKey)
            cursors[bestList] += 1
            remaining -= 1
        }
        return SelectionResult(ids: mergedIds, cutoff: cutoff)
    }

    /// Exact fp32 distances for gathered rows. Formulas match
    /// `ParallelFlatScan.selectTopK` so both paths agree bit-for-bit on ties.
    private static func rescore(
        ids: [UInt32],
        corpus: UnsafePointer<Float>,
        query: UnsafePointer<Float>,
        norms: UnsafePointer<Float>?,
        queryNormSq: Float,
        dim: Int,
        metric: Metric
    ) -> [ParallelFlatScan.Entry] {
        guard !ids.isEmpty else { return [] }
        var dots = [Float](repeating: 0, count: ids.count)
        ids.withUnsafeBufferPointer { idsBuffer in
            dots.withUnsafeMutableBufferPointer { dotsBuffer in
                mans_f32_dot_rows_gather(
                    corpus,
                    query,
                    idsBuffer.baseAddress!,
                    Int64(ids.count),
                    Int64(dim),
                    dotsBuffer.baseAddress!
                )
            }
        }

        var entries = [ParallelFlatScan.Entry]()
        entries.reserveCapacity(ids.count)
        switch metric {
        case .innerProduct:
            for index in 0..<ids.count {
                entries.append(ParallelFlatScan.Entry(distance: -dots[index], id: ids[index]))
            }
        case .l2:
            let normsBase = norms!
            for index in 0..<ids.count {
                let d = max(0, queryNormSq - 2 * dots[index] + normsBase[Int(ids[index])])
                entries.append(ParallelFlatScan.Entry(distance: d, id: ids[index]))
            }
        case .cosine:
            let normsBase = norms!
            for index in 0..<ids.count {
                let denom = (queryNormSq * normsBase[Int(ids[index])]).squareRoot()
                let d = denom < 1e-10 ? 1.0 : 1.0 - dots[index] / denom
                entries.append(ParallelFlatScan.Entry(distance: d, id: ids[index]))
            }
        case .hamming:
            break
        }
        entries.sort { $0.distance < $1.distance }
        return entries
    }
}

/// Immutable bridge carrying scan inputs into the concurrent passes.
private final class ScanContext: @unchecked Sendable {
    let queryCodes: [Int8]
    let codes: UnsafeMutablePointer<Int8>
    let uPlane: UnsafeMutablePointer<Float>
    let wPlane: UnsafeMutablePointer<Float>
    let normSqPlane: UnsafeMutablePointer<Float>
    let invSqrtPlane: UnsafeMutablePointer<Float>
    let dim: Int
    let metric: Metric
    let slices: Int
    let chunkSize: Int

    // Query-side coefficients (see search() doc comment).
    let qsDot: Float
    let qsW: Float
    let qsQabsHalf: Float
    let cosFDot: Float
    let cosFH: Float
    let cosFW: Float
    let invSqrtQueryNorm: Float
    let queryNormSq: Float
    let queryNormSqD: Double

    init(
        queryCodes: [Int8],
        codes: UnsafeMutablePointer<Int8>,
        uPlane: UnsafeMutablePointer<Float>,
        wPlane: UnsafeMutablePointer<Float>,
        normSqPlane: UnsafeMutablePointer<Float>,
        invSqrtPlane: UnsafeMutablePointer<Float>,
        dim: Int,
        metric: Metric,
        slices: Int,
        chunkSize: Int,
        qsDot: Float,
        qsW: Float,
        qsQabsHalf: Float,
        queryNormSq: Float,
        cosFDot: Float,
        cosFH: Float,
        cosFW: Float,
        invSqrtQueryNorm: Float,
        queryNormSqD: Double
    ) {
        self.queryCodes = queryCodes
        self.codes = codes
        self.uPlane = uPlane
        self.wPlane = wPlane
        self.normSqPlane = normSqPlane
        self.invSqrtPlane = invSqrtPlane
        self.dim = dim
        self.metric = metric
        self.slices = slices
        self.chunkSize = chunkSize
        self.qsDot = qsDot
        self.qsW = qsW
        self.qsQabsHalf = qsQabsHalf
        self.queryNormSq = queryNormSq
        self.cosFDot = cosFDot
        self.cosFH = cosFH
        self.cosFW = cosFW
        self.invSqrtQueryNorm = invSqrtQueryNorm
        self.queryNormSqD = queryNormSqD
    }
}
