import Accelerate
import Foundation
import MetalANNS
import MetalANNSCore

/// Measures build/insert/delete/persistence costs at the public `GraphIndex`
/// API and re-validates recall after every mutation against an independent
/// harness-side BLAS brute-force ground truth (this scan is deliberately
/// separate from the library's own search implementation).
struct UpdateCostBench {
    struct Results: Sendable {
        var vectorCount: Int
        var dim: Int
        var metric: Metric
        var updateCount: Int
        var buildTimeMs: Double
        var insertTotalMs: Double
        var insertMeanMs: Double
        var insertQPS: Double
        var deleteCount: Int
        var deleteTotalMs: Double
        var deleteMeanMs: Double
        var compactMs: Double
        var saveMs: Double
        var loadMs: Double
        var savedBytes: Int
        var recallAfterInserts: Double
        var recallAfterDeletes: Double
        var recallAfterCompact: Double
        var recallAfterReload: Double
    }

    /// The corpus as the caller's key-space sees it: appended rows carry their
    /// external key, deletes remove by key, re-inserts append a fresh row with
    /// the same key (mirroring GraphIndex soft-delete + append semantics).
    private struct LiveCorpus {
        var rows: [(externalID: String, vector: [Float])] = []

        mutating func append(id: String, vector: [Float]) {
            rows.append((id, vector))
        }

        mutating func remove(ids: Set<String>) {
            rows.removeAll { ids.contains($0.externalID) }
        }
    }

    static func run(
        trainVectors: [[Float]],
        queries: [[Float]],
        metric: Metric,
        configuration: IndexConfiguration,
        updateCount: Int
    ) async throws -> Results {
        let baseCount = trainVectors.count
        let dim = trainVectors[0].count
        let updates = max(1, min(updateCount, baseCount / 2))
        let ids = (0..<baseCount).map { "v_\($0)" }

        let index = GraphIndex(configuration: configuration)
        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: trainVectors, ids: ids)
        let buildEnd = DispatchTime.now().uptimeNanoseconds
        let buildTimeMs = Double(buildEnd - buildStart) / 1_000_000.0

        let mutationVectors = BenchmarkRunner.makeSeededMutationVectors(
            count: updates,
            dim: dim,
            seed: 9_181_001
        )

        // -- Appends -----------------------------------------------------
        var insertElapsed: UInt64 = 0
        for offset in 0..<updates {
            let start = DispatchTime.now().uptimeNanoseconds
            try await index.insert(mutationVectors[offset], id: "u_\(offset)")
            insertElapsed &+= DispatchTime.now().uptimeNanoseconds &- start
        }

        var live = LiveCorpus()
        live.rows.reserveCapacity(baseCount + updates * 2)
        for (offset, id) in ids.enumerated() {
            live.append(id: id, vector: trainVectors[offset])
        }
        for offset in 0..<updates {
            live.append(id: "u_\(offset)", vector: mutationVectors[offset])
        }

        let recallAfterInserts = try await recall(
            index: index, queries: queries, corpus: live, metric: metric
        )

        // -- Deletes (half original rows, half freshly appended) ----------
        var deletedIDs: [String] = []
        for offset in 0..<(updates / 2) {
            deletedIDs.append("v_\(offset * 3)")
        }
        for offset in 0..<(updates - updates / 2) {
            deletedIDs.append("u_\(offset * 2)")
        }
        var deleteElapsed: UInt64 = 0
        for id in deletedIDs {
            let start = DispatchTime.now().uptimeNanoseconds
            try await index.delete(id: id)
            deleteElapsed &+= DispatchTime.now().uptimeNanoseconds &- start
        }
        live.remove(ids: Set(deletedIDs))

        let recallAfterDeletes = try await recall(
            index: index, queries: queries, corpus: live, metric: metric
        )

        // -- Compact (reclaims soft-deleted slots; rebuilds storage) -------
        let compactStart = DispatchTime.now().uptimeNanoseconds
        try await index.compact()
        let compactMs = Double(DispatchTime.now().uptimeNanoseconds &- compactStart) / 1_000_000.0

        let recallAfterCompact = try await recall(
            index: index, queries: queries, corpus: live, metric: metric
        )

        // -- Persistence ---------------------------------------------------
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-update-bench-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let saveStart = DispatchTime.now().uptimeNanoseconds
        try await index.save(to: tempURL)
        let saveEnd = DispatchTime.now().uptimeNanoseconds
        let savedBytes =
            (try? FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int) ?? 0
        let loaded = try await GraphIndex.load(from: tempURL)
        let loadEnd = DispatchTime.now().uptimeNanoseconds

        let recallAfterReload = try await recall(
            index: loaded, queries: queries, corpus: live, metric: metric
        )

        let insertTotalMs = Double(insertElapsed) / 1_000_000.0
        let deleteTotalMs = Double(deleteElapsed) / 1_000_000.0

        return Results(
            vectorCount: baseCount,
            dim: dim,
            metric: metric,
            updateCount: updates,
            buildTimeMs: buildTimeMs,
            insertTotalMs: insertTotalMs,
            insertMeanMs: insertTotalMs / Double(updates),
            insertQPS: Double(updates) / (insertTotalMs / 1_000.0),
            deleteCount: deletedIDs.count,
            deleteTotalMs: deleteTotalMs,
            deleteMeanMs: deleteTotalMs / Double(deletedIDs.count),
            compactMs: compactMs,
            saveMs: Double(saveEnd - saveStart) / 1_000_000.0,
            loadMs: Double(loadEnd - saveEnd) / 1_000_000.0,
            savedBytes: savedBytes,
            recallAfterInserts: recallAfterInserts,
            recallAfterDeletes: recallAfterDeletes,
            recallAfterCompact: recallAfterCompact,
            recallAfterReload: recallAfterReload
        )
    }

    /// Mean recall@10 of `index.search` vs the harness-side BLAS ground truth
    /// over the live corpus.
    private static func recall(
        index: GraphIndex,
        queries: [[Float]],
        corpus: LiveCorpus,
        metric: Metric
    ) async throws -> Double {
        let k = 10
        var total = 0.0

        for query in queries {
            let expected = Set(
                blasTopKExternalIDs(query: query, corpus: corpus.rows, k: k, metric: metric)
            )
            guard !expected.isEmpty else { continue }
            let results = try await index.search(query: query, k: k)
            let got = Set(results.map(\.id))
            total += Double(got.intersection(expected).count) / Double(expected.count)
        }
        return total / Double(queries.count)
    }

    /// Independent top-K via BLAS: dots from `cblas_sgemv`, corpus norms from a
    /// squared-values `sgemv`, bounded selection by insertion into a sorted
    /// prefix. Distinct from the library's heap selector on purpose.
    private static func blasTopKExternalIDs(
        query: [Float],
        corpus: [(externalID: String, vector: [Float])],
        k: Int,
        metric: Metric
    ) -> [String] {
        let count = corpus.count
        guard count > 0, query.count == corpus[0].vector.count else { return [] }
        let dim = query.count
        let effectiveK = min(k, count)

        var flat = [Float](repeating: 0, count: count * dim)
        var dots = [Float](repeating: 0, count: count)
        var normsSq = [Float](repeating: 0, count: count)
        var queryNormSq: Float = 0

        flat.withUnsafeMutableBufferPointer { flatBuffer in
            guard let corpusBase = flatBuffer.baseAddress else { return }
            for row in 0..<count {
                corpus[row].vector.withUnsafeBufferPointer { source in
                    source.baseAddress.map {
                        corpusBase.advanced(by: row * dim).update(from: $0, count: dim)
                    }
                }
            }

            dots.withUnsafeMutableBufferPointer { dotsBuffer in
                normsSq.withUnsafeMutableBufferPointer { normsBuffer in
                    query.withUnsafeBufferPointer { queryBuffer in
                        guard let q = queryBuffer.baseAddress,
                            let dotsBase = dotsBuffer.baseAddress,
                            let normsBase = normsBuffer.baseAddress
                        else { return }

                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans, Int32(count), Int32(dim),
                            1.0, corpusBase, Int32(dim), q, 1, 0.0, dotsBase, 1
                        )
                        if metric == .cosine || metric == .l2 {
                            for value in query { queryNormSq += value * value }
                        }
                        if metric == .cosine {
                            var squared = [Float](repeating: 0, count: count * dim)
                            squared.withUnsafeMutableBufferPointer { squaredBuffer in
                                guard let squaredBase = squaredBuffer.baseAddress else { return }
                                vDSP_vsq(corpusBase, 1, squaredBase, 1, vDSP_Length(count * dim))
                                let ones = [Float](repeating: 1, count: dim)
                                ones.withUnsafeBufferPointer { onesBuffer in
                                    guard let onesBase = onesBuffer.baseAddress else { return }
                                    cblas_sgemv(
                                        CblasRowMajor, CblasNoTrans, Int32(count), Int32(dim),
                                        1.0, squaredBase, Int32(dim), onesBase, 1, 0.0, normsBase, 1
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        var bestDistances = [Float](repeating: .infinity, count: effectiveK)
        var bestIDs = [String?](repeating: nil, count: effectiveK)

        for row in 0..<count {
            let distance: Float
            switch metric {
            case .cosine:
                let denom = (queryNormSq * normsSq[row]).squareRoot()
                distance = denom < 1e-10 ? 1.0 : (1.0 - dots[row] / denom)
            case .l2:
                var normV: Float = 0
                let v = corpus[row].vector
                for value in v { normV += value * value }
                distance = max(0, queryNormSq - 2.0 * dots[row] + normV)
            case .innerProduct:
                distance = -dots[row]
            case .hamming:
                distance = .infinity
            }

            var slot = effectiveK - 1
            if distance.isNaN || slot < 0 || distance >= bestDistances[slot] { continue }
            while slot > 0, distance < bestDistances[slot - 1] {
                bestDistances[slot] = bestDistances[slot - 1]
                bestIDs[slot] = bestIDs[slot - 1]
                slot -= 1
            }
            bestDistances[slot] = distance
            bestIDs[slot] = corpus[row].externalID
        }

        return bestIDs.compactMap { $0 }
    }
}

extension BenchmarkRunner {
    /// Deterministic mutation vectors for the update-cost benchmark
    /// (sin/cos progression, disjoint seed range from the corpus/queries).
    static func makeSeededMutationVectors(count: Int, dim: Int, seed: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seed) * dim + col)
                return sin(i * 0.137) + cos(i * 0.089)
            }
        }
    }
}
