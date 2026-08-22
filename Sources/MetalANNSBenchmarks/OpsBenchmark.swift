import Foundation
import MetalANNS
import MetalANNSCore

/// Measures index lifecycle costs beyond query throughput: incremental
/// inserts, deletes, compaction, serialization, and reload. Each stage is
/// followed by a seeded exact-recall spot check so update correctness is
/// verified alongside cost.
enum OpsBenchmark {
    struct Config {
        var vectorCount: Int = 10_000
        var insertCount: Int = 1_000
        var deleteCount: Int = 500
        var dim: Int = 128
        var metric: Metric = .cosine
        var seed: Int = 42
    }

    struct StageResult: Sendable {
        let name: String
        let totalMs: Double
        let perOpMs: Double?
        let extra: String

        init(name: String, totalMs: Double, opCount: Int? = nil, extra: String = "") {
            self.name = name
            self.totalMs = totalMs
            if let opCount, opCount > 0 {
                self.perOpMs = totalMs / Double(opCount)
            } else {
                self.perOpMs = nil
            }
            self.extra = extra
        }
    }

    static func run(config: Config) async throws -> [StageResult] {
        let vectors = makeVectors(count: config.vectorCount, dim: config.dim, seedOffset: config.seed)
        let insertedVectors = makeVectors(
            count: config.insertCount, dim: config.dim, seedOffset: config.seed &+ 500_000)
        let ids = (0..<config.vectorCount).map { "v_\($0)" }
        let insertedIDs = (0..<config.insertCount).map { "new_\($0)" }

        var stages: [StageResult] = []

        let index = GraphIndex(
            configuration: IndexConfiguration(metric: config.metric))

        // 1. Build
        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: vectors, ids: ids)
        let buildMs = elapsedMs(buildStart)
        stages.append(StageResult(name: "build", totalMs: buildMs, opCount: config.vectorCount))

        // Recall baseline before updates.
        let beforeRecall = try await spotRecall(index: index, corpus: vectors, config: config)

        // 2. Incremental inserts
        let insertStart = DispatchTime.now().uptimeNanoseconds
        try await index.batchInsert(insertedVectors, ids: insertedIDs)
        let insertMs = elapsedMs(insertStart)
        stages.append(StageResult(name: "batchInsert", totalMs: insertMs, opCount: config.insertCount))

        let afterInsertRecall = try await spotRecall(
            index: index, corpus: vectors + insertedVectors, config: config)

        // 3. Deletes (soft)
        let deleteTargets = Array(ids.prefix(config.deleteCount))
        let deleteStart = DispatchTime.now().uptimeNanoseconds
        for id in deleteTargets {
            try await index.delete(id: id)
        }
        let deleteMs = elapsedMs(deleteStart)
        stages.append(StageResult(name: "delete", totalMs: deleteMs, opCount: config.deleteCount))

        // Recall after deletes must exclude deleted rows. Internal IDs are
        // stable across soft deletes, so compute truth over the full corpus
        // and drop deleted indices from each candidate list.
        let deletedInternalIDs = Set((0..<config.deleteCount).map(UInt32.init))
        let surviving = Array(vectors.dropFirst(config.deleteCount)) + insertedVectors
        let afterDeleteRecall = try await spotRecall(
            index: index, corpus: vectors + insertedVectors,
            excludedInternalIDs: deletedInternalIDs, config: config)

        // 4. Compaction
        let compactStart = DispatchTime.now().uptimeNanoseconds
        try await index.compact()
        let compactMs = elapsedMs(compactStart)
        stages.append(StageResult(name: "compact", totalMs: compactMs))
        let afterCompactCount = await index.count
        let afterCompactRecall = try await spotRecall(index: index, corpus: surviving, config: config)

        // 5. Serialize (mmap-compatible format so both loaders can read it)
        let saveURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
            "metalanns-ops-\(UUID().uuidString).annindex")
        let saveStart = DispatchTime.now().uptimeNanoseconds
        try await index.saveMmapCompatible(to: saveURL)
        let saveMs = elapsedMs(saveStart)
        let attrs = (try? FileManager.default.attributesOfItem(atPath: saveURL.path)) ?? [:]
        let fileBytes = attrs[.size] as? Int ?? 0
        let fileSizeMB = Double(fileBytes) / (1024 * 1024)
        stages.append(StageResult(name: "save", totalMs: saveMs, extra: String(format: "%.1f MB", fileSizeMB)))

        // 6. Reload (full memory)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let reloaded = try await GraphIndex.load(from: saveURL)
        let loadMs = elapsedMs(loadStart)
        stages.append(StageResult(name: "load(fullMemory)", totalMs: loadMs))
        let reloadedCount = await reloaded.count
        let reloadedRecall = try await spotRecall(index: reloaded, corpus: surviving, config: config)

        // 7. Reload (mmap zero-copy)
        let mmapStart = DispatchTime.now().uptimeNanoseconds
        let mmapped = try await GraphIndex.loadMmap(from: saveURL)
        let mmapMs = elapsedMs(mmapStart)
        stages.append(StageResult(name: "loadMmap", totalMs: mmapMs))
        let mmapRecall = try await spotRecall(index: mmapped, corpus: surviving, config: config)

        try? FileManager.default.removeItem(at: saveURL)

        print("Recall@10 spot checks:")
        print("  before updates:    \(String(format: "%.3f", beforeRecall))")
        print("  after inserts:     \(String(format: "%.3f", afterInsertRecall))")
        print("  after deletes:     \(String(format: "%.3f", afterDeleteRecall))")
        print(
            "  after compact:     \(String(format: "%.3f", afterCompactRecall)) (count=\(afterCompactCount), expected=\(surviving.count))"
        )
        print("  reloaded:          \(String(format: "%.3f", reloadedRecall)) (count=\(reloadedCount))")
        print("  reloaded (mmap):   \(String(format: "%.3f", mmapRecall))")

        return stages
    }

    /// Exact-recall@10 spot check against brute force over the provided corpus.
    private static func spotRecall(
        index: GraphIndex,
        corpus: [[Float]],
        excludedInternalIDs: Set<UInt32> = [],
        config: Config
    ) async throws -> Double {
        let queries = makeVectors(count: 8, dim: config.dim, seedOffset: config.seed &+ 900_001)
        let k = 10
        var hits = 0
        var total = 0
        for query in queries {
            let truth = Set(
                bruteForceTopK(query: query, vectors: corpus, k: k + excludedInternalIDs.count, metric: config.metric)
                    .filter { !excludedInternalIDs.contains($0) }
                    .prefix(k))
            let results = try await index.search(query: query, k: k)
            let got = Set(results.prefix(k).map(\.internalID))
            hits += got.intersection(truth).count
            total += k
        }
        return Double(hits) / Double(max(total, 1))
    }

    private static func elapsedMs(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds &- start) / 1_000_000.0
    }

    private static func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    private static func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> [UInt32] {
        let topK = max(1, min(k, vectors.count))
        return
            vectors
            .enumerated()
            .map { idx, vector -> (id: UInt32, distance: Float) in
                (UInt32(idx), distance(query: query, vector: vector, metric: metric))
            }
            .sorted { $0.distance < $1.distance }
            .prefix(topK)
            .map(\.id)
    }

    private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        case .hamming:
            var mismatches = 0
            for d in 0..<query.count where query[d] != vector[d] {
                mismatches += 1
            }
            return Float(mismatches)
        }
    }

    static func renderTable(_ stages: [StageResult]) -> String {
        var lines: [String] = []
        lines.append(
            String(
                format: "%-18@ %12@ %14@  %@", "stage" as NSString, "total(ms)" as NSString,
                "per-op(ms)" as NSString, "extra" as NSString))
        lines.append(String(repeating: "-", count: 62))
        for stage in stages {
            let perOp = stage.perOpMs.map { String(format: "%.4f", $0) } ?? "-"
            lines.append(
                String(
                    format: "%-18@ %12.2f %14@  %@",
                    stage.name as NSString, stage.totalMs, perOp as NSString, stage.extra as NSString))
        }
        return lines.joined(separator: "\n")
    }
}
