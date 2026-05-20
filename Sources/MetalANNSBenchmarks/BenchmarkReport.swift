import Foundation

public struct BenchmarkReport: Sendable {
    public struct Row: Sendable {
        public var label: String
        public var recallAt10: Double
        public var qps: Double
        public var buildTimeMs: Double
        public var p50Ms: Double
        public var p95Ms: Double
        public var p99Ms: Double
        public var recallAt1: Double
        public var recallAt100: Double
        public var queryCount: Int
        public var avgQueryMs: Double
        public var maxQueryMs: Double
        public var requestedK: Int
        public var effectiveK: Int
        public var operation: String
        public var operationTimeMs: Double
        public var estimatedBackendPath: String
        public var rssBeforeBytes: UInt64
        public var rssAfterBytes: UInt64
        public var rssDeltaBytes: Int64
        public var fileSizeBytes: UInt64

        public init(
            label: String,
            recallAt10: Double,
            qps: Double,
            buildTimeMs: Double,
            p50Ms: Double,
            p95Ms: Double,
            p99Ms: Double,
            recallAt1: Double = 0,
            recallAt100: Double = 0,
            queryCount: Int = 0,
            avgQueryMs: Double = 0,
            maxQueryMs: Double = 0,
            requestedK: Int = 0,
            effectiveK: Int = 0,
            operation: String = "",
            operationTimeMs: Double = 0,
            estimatedBackendPath: String = "",
            rssBeforeBytes: UInt64 = 0,
            rssAfterBytes: UInt64 = 0,
            rssDeltaBytes: Int64 = 0,
            fileSizeBytes: UInt64 = 0
        ) {
            self.label = label
            self.recallAt10 = recallAt10
            self.qps = qps
            self.buildTimeMs = buildTimeMs
            self.p50Ms = p50Ms
            self.p95Ms = p95Ms
            self.p99Ms = p99Ms
            self.recallAt1 = recallAt1
            self.recallAt100 = recallAt100
            self.queryCount = queryCount
            self.avgQueryMs = avgQueryMs
            self.maxQueryMs = maxQueryMs
            self.requestedK = requestedK
            self.effectiveK = effectiveK
            self.operation = operation
            self.operationTimeMs = operationTimeMs
            self.estimatedBackendPath = estimatedBackendPath
            self.rssBeforeBytes = rssBeforeBytes
            self.rssAfterBytes = rssAfterBytes
            self.rssDeltaBytes = rssDeltaBytes
            self.fileSizeBytes = fileSizeBytes
        }
    }

    public var rows: [Row]
    public var datasetLabel: String
    public var metadata: [String: String]
    public var generatedAt: String

    public init(
        rows: [Row],
        datasetLabel: String,
        metadata: [String: String] = [:],
        generatedAt: String = ISO8601DateFormatter().string(from: Date())
    ) {
        self.rows = rows
        self.datasetLabel = datasetLabel
        self.metadata = metadata
        self.generatedAt = generatedAt
    }

    public func renderTable() -> String {
        var lines: [String] = []
        lines.append(
            padRight("label", to: 16)
                + " "
                + padLeft("recall@10", to: 10)
                + " "
                + padLeft("QPS", to: 8)
                + " "
                + padLeft("buildMs", to: 9)
                + " "
                + padLeft("p50ms", to: 7)
                + " "
                + padLeft("p95ms", to: 7)
                + " "
                + padLeft("p99ms", to: 7)
                + " "
                + padLeft("opMs", to: 7)
        )
        lines.append(String(repeating: "-", count: 82))

        for row in rows {
            lines.append(
                padRight(row.label, to: 16)
                    + " "
                    + padLeft(String(format: "%.3f", row.recallAt10), to: 10)
                    + " "
                    + padLeft(String(format: "%.0f", row.qps), to: 8)
                    + " "
                    + padLeft(String(format: "%.1f", row.buildTimeMs), to: 9)
                    + " "
                    + padLeft(String(format: "%.2f", row.p50Ms), to: 7)
                    + " "
                    + padLeft(String(format: "%.2f", row.p95Ms), to: 7)
                    + " "
                    + padLeft(String(format: "%.2f", row.p99Ms), to: 7)
                    + " "
                    + padLeft(String(format: "%.2f", row.operationTimeMs), to: 7)
            )
        }

        return lines.joined(separator: "\n")
    }

    public func renderCSV() -> String {
        var lines = ["label,recall@1,recall@10,recall@100,qps,buildTimeMs,p50ms,p95ms,p99ms,queryCount,avgQueryMs,maxQueryMs,requestedK,effectiveK,operation,operationTimeMs,estimatedBackendPath,rssBeforeBytes,rssAfterBytes,rssDeltaBytes,fileSizeBytes"]
        for row in rows {
            lines.append(
                [
                    csvEscape(row.label),
                    String(format: "%.6f", row.recallAt1),
                    String(format: "%.6f", row.recallAt10),
                    String(format: "%.6f", row.recallAt100),
                    String(format: "%.6f", row.qps),
                    String(format: "%.6f", row.buildTimeMs),
                    String(format: "%.6f", row.p50Ms),
                    String(format: "%.6f", row.p95Ms),
                    String(format: "%.6f", row.p99Ms),
                    String(row.queryCount),
                    String(format: "%.6f", row.avgQueryMs),
                    String(format: "%.6f", row.maxQueryMs),
                    String(row.requestedK),
                    String(row.effectiveK),
                    csvEscape(row.operation),
                    String(format: "%.6f", row.operationTimeMs),
                    csvEscape(row.estimatedBackendPath),
                    String(row.rssBeforeBytes),
                    String(row.rssAfterBytes),
                    String(row.rssDeltaBytes),
                    String(row.fileSizeBytes)
                ]
                .joined(separator: ",")
            )
        }
        return lines.joined(separator: "\n") + "\n"
    }

    public func renderJSON() -> String {
        let payload: [String: Any] = [
            "datasetLabel": datasetLabel,
            "generatedAt": generatedAt,
            "metadata": metadata,
            "rows": rows.map { row in
                [
                    "label": row.label,
                    "recallAt10": row.recallAt10,
                    "qps": row.qps,
                    "buildTimeMs": row.buildTimeMs,
                    "p50Ms": row.p50Ms,
                    "p95Ms": row.p95Ms,
                    "p99Ms": row.p99Ms,
                    "recallAt1": row.recallAt1,
                    "recallAt100": row.recallAt100,
                    "queryCount": row.queryCount,
                    "avgQueryMs": row.avgQueryMs,
                    "maxQueryMs": row.maxQueryMs,
                    "requestedK": row.requestedK,
                    "effectiveK": row.effectiveK,
                    "operation": row.operation,
                    "operationTimeMs": row.operationTimeMs,
                    "estimatedBackendPath": row.estimatedBackendPath,
                    "rssBeforeBytes": row.rssBeforeBytes,
                    "rssAfterBytes": row.rssAfterBytes,
                    "rssDeltaBytes": row.rssDeltaBytes,
                    "fileSizeBytes": row.fileSizeBytes
                ]
            }
        ]

        let jsonData = try? JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]
        )
        return String(data: jsonData ?? Data("{}".utf8), encoding: .utf8) ?? "{}"
    }

    public func saveCSV(to path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try renderCSV().write(to: url, atomically: true, encoding: .utf8)
    }

    public func saveJSON(to path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try renderJSON().write(to: url, atomically: true, encoding: .utf8)
    }

    public func paretoFrontier() -> [Row] {
        rows.filter { candidate in
            !rows.contains { other in
                guard other.label != candidate.label else {
                    return false
                }
                let dominates = other.recallAt10 >= candidate.recallAt10
                    && other.qps >= candidate.qps
                    && (other.recallAt10 > candidate.recallAt10 || other.qps > candidate.qps)
                return dominates
            }
        }
        .sorted { lhs, rhs in
            if lhs.recallAt10 == rhs.recallAt10 {
                return lhs.qps > rhs.qps
            }
            return lhs.recallAt10 > rhs.recallAt10
        }
    }

    private func csvEscape(_ text: String) -> String {
        if text.contains(",") || text.contains("\"") || text.contains("\n") {
            let escaped = text.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
        }
        return text
    }

    private func padLeft(_ text: String, to width: Int) -> String {
        if text.count >= width {
            return text
        }
        return String(repeating: " ", count: width - text.count) + text
    }

    private func padRight(_ text: String, to width: Int) -> String {
        if text.count >= width {
            return text
        }
        return text + String(repeating: " ", count: width - text.count)
    }
}
