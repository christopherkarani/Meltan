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
        public var p90Ms: Double
        public var p999Ms: Double
        public var stdDevMs: Double
        public var minMs: Double
        public var indexResidentMB: Double = 0
        public var peakResidentMB: Double = 0
        public var concurrency: Int = 1
        public var firstQueryMs: Double = 0
        public var warmSteadyMeanMs: Double = 0
        public var backendLabel: String = ""

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
            p90Ms: Double = 0,
            p999Ms: Double = 0,
            stdDevMs: Double = 0,
            minMs: Double = 0,
            indexResidentMB: Double = 0,
            peakResidentMB: Double = 0,
            concurrency: Int = 1,
            firstQueryMs: Double = 0,
            warmSteadyMeanMs: Double = 0,
            backendLabel: String = ""
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
            self.p90Ms = p90Ms
            self.p999Ms = p999Ms
            self.stdDevMs = stdDevMs
            self.minMs = minMs
            self.indexResidentMB = indexResidentMB
            self.peakResidentMB = peakResidentMB
            self.concurrency = concurrency
            self.firstQueryMs = firstQueryMs
            self.warmSteadyMeanMs = warmSteadyMeanMs
            self.backendLabel = backendLabel
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
        let showConcurrency = rows.contains { $0.concurrency != 1 }

        var header = padRight("label", to: 16)
            + " "
            + padLeft("recall@10", to: 10)
            + " "
            + padLeft("QPS", to: 8)
            + " "
            + padLeft("buildMs", to: 9)
            + " "
            + padLeft("p50ms", to: 7)
            + " "
            + padLeft("p90ms", to: 7)
            + " "
            + padLeft("p95ms", to: 7)
            + " "
            + padLeft("p99ms", to: 7)
            + " "
            + padLeft("p999ms", to: 8)
        if showConcurrency {
            header += " " + padLeft("conc", to: 5)
        }
        lines.append(header)
        lines.append(String(repeating: "-", count: showConcurrency ? 97 : 91))

        for row in rows {
            var line = padRight(row.label, to: 16)
                + " "
                + padLeft(String(format: "%.3f", row.recallAt10), to: 10)
                + " "
                + padLeft(String(format: "%.0f", row.qps), to: 8)
                + " "
                + padLeft(String(format: "%.1f", row.buildTimeMs), to: 9)
                + " "
                + padLeft(String(format: "%.2f", row.p50Ms), to: 7)
                + " "
                + padLeft(String(format: "%.2f", row.p90Ms), to: 7)
                + " "
                + padLeft(String(format: "%.2f", row.p95Ms), to: 7)
                + " "
                + padLeft(String(format: "%.2f", row.p99Ms), to: 7)
                + " "
                + padLeft(String(format: "%.2f", row.p999Ms), to: 8)
            if showConcurrency {
                line += " " + padLeft(String(row.concurrency), to: 5)
            }
            lines.append(line)
        }

        return lines.joined(separator: "\n")
    }

    public func renderCSV() -> String {
        var lines = ["label,recall@10,qps,buildTimeMs,p50ms,p90ms,p95ms,p99ms,p999ms,minMs,stdDevMs,avgQueryMs,maxQueryMs,recall@1,recall@100,queryCount,indexResidentMB,peakResidentMB,concurrency"]
        for row in rows {
            lines.append(
                [
                    csvEscape(row.label),
                    String(format: "%.6f", row.recallAt10),
                    String(format: "%.6f", row.qps),
                    String(format: "%.6f", row.buildTimeMs),
                    String(format: "%.6f", row.p50Ms),
                    String(format: "%.6f", row.p90Ms),
                    String(format: "%.6f", row.p95Ms),
                    String(format: "%.6f", row.p99Ms),
                    String(format: "%.6f", row.p999Ms),
                    String(format: "%.6f", row.minMs),
                    String(format: "%.6f", row.stdDevMs),
                    String(format: "%.6f", row.avgQueryMs),
                    String(format: "%.6f", row.maxQueryMs),
                    String(format: "%.6f", row.recallAt1),
                    String(format: "%.6f", row.recallAt100),
                    String(row.queryCount),
                    String(format: "%.6f", row.indexResidentMB),
                    String(format: "%.6f", row.peakResidentMB),
                    String(row.concurrency)
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
                    "p90Ms": row.p90Ms,
                    "p95Ms": row.p95Ms,
                    "p99Ms": row.p99Ms,
                    "p999Ms": row.p999Ms,
                    "stdDevMs": row.stdDevMs,
                    "minMs": row.minMs,
                    "recallAt1": row.recallAt1,
                    "recallAt100": row.recallAt100,
                    "queryCount": row.queryCount,
                    "avgQueryMs": row.avgQueryMs,
                    "maxQueryMs": row.maxQueryMs,
                    "indexResidentMB": row.indexResidentMB,
                    "peakResidentMB": row.peakResidentMB,
                    "concurrency": row.concurrency,
                    "firstQueryMs": row.firstQueryMs,
                    "warmSteadyMeanMs": row.warmSteadyMeanMs,
                    "backendLabel": row.backendLabel
                ]
            }
        ]

        let jsonData = try? JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]
        )
        return String(data: jsonData ?? Data("{}".utf8), encoding: .utf8) ?? "{}"
    }

    public func renderParetoChart(width: Int = 60, height: Int = 16) -> String {
        guard !rows.isEmpty else {
            return "(no data)"
        }

        let safeWidth = max(20, width)
        let safeHeight = max(6, height)

        let frontierLabels = Set(paretoFrontier().map { $0.label })

        let qpsValues = rows.map { Swift.max($0.qps, 1e-9) }
        let logQps = qpsValues.map { log10($0) }
        guard let minLogQps = logQps.min(), let maxLogQps = logQps.max() else {
            return "(no data)"
        }
        let logRange = maxLogQps - minLogQps
        let usableLogRange = logRange > 0 ? logRange : 1.0

        let plotWidth = safeWidth - 8
        let plotHeight = safeHeight - 3
        guard plotWidth > 0, plotHeight > 0 else {
            return "(chart too small)"
        }

        var grid = Array(repeating: Array(repeating: Character(" "), count: plotWidth), count: plotHeight)

        for (idx, row) in rows.enumerated() {
            let recall = Swift.min(Swift.max(row.recallAt10, 0), 1)
            let xFraction = recall
            let yFraction = (logQps[idx] - minLogQps) / usableLogRange
            let xPos = Swift.min(plotWidth - 1, Swift.max(0, Int(xFraction * Double(plotWidth - 1))))
            let yPosFromTop = Swift.min(plotHeight - 1, Swift.max(0, plotHeight - 1 - Int(yFraction * Double(plotHeight - 1))))
            let isFrontier = frontierLabels.contains(row.label)
            let marker: Character = isFrontier ? "*" : "."
            let existing = grid[yPosFromTop][xPos]
            if existing == " " || (marker == "*" && existing == ".") {
                grid[yPosFromTop][xPos] = marker
            }
        }

        var lines: [String] = []
        lines.append("QPS (log10) vs recall@10  -- '*' = Pareto frontier, '.' = dominated")
        for (rowIndex, line) in grid.enumerated() {
            let logValue = maxLogQps - (Double(rowIndex) / Double(max(1, plotHeight - 1))) * usableLogRange
            let qpsLabel = String(format: "%6.0f", pow(10.0, logValue))
            lines.append("\(qpsLabel) | " + String(line))
        }
        let axis = String(repeating: "-", count: plotWidth)
        lines.append("       +" + axis)
        let leftPadWidth = max(3, 9)
        let midPadWidth = max(3, plotWidth / 2 - 2)
        let rightPadWidth = max(3, plotWidth / 2 + 1)
        let xAxisLabels = padLeft("0.0", to: leftPadWidth)
            + padLeft("0.5", to: midPadWidth)
            + padLeft("1.0", to: rightPadWidth)
        lines.append(xAxisLabels)

        return lines.joined(separator: "\n")
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
