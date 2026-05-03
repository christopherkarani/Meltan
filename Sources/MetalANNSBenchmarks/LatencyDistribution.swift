import Foundation

public struct LatencyDistribution: Sendable {
    public let countSamples: Int
    public let meanMs: Double
    public let stdDevMs: Double
    public let minMs: Double
    public let maxMs: Double
    public let p50Ms: Double
    public let p90Ms: Double
    public let p95Ms: Double
    public let p99Ms: Double
    public let p999Ms: Double
    public let histogramBuckets: [HistogramBucket]

    public struct HistogramBucket: Sendable {
        public let lowerMs: Double
        public let upperMs: Double
        public let count: Int

        public init(lowerMs: Double, upperMs: Double, count: Int) {
            self.lowerMs = lowerMs
            self.upperMs = upperMs
            self.count = count
        }
    }

    public init(
        countSamples: Int,
        meanMs: Double,
        stdDevMs: Double,
        minMs: Double,
        maxMs: Double,
        p50Ms: Double,
        p90Ms: Double,
        p95Ms: Double,
        p99Ms: Double,
        p999Ms: Double,
        histogramBuckets: [HistogramBucket]
    ) {
        self.countSamples = countSamples
        self.meanMs = meanMs
        self.stdDevMs = stdDevMs
        self.minMs = minMs
        self.maxMs = maxMs
        self.p50Ms = p50Ms
        self.p90Ms = p90Ms
        self.p95Ms = p95Ms
        self.p99Ms = p99Ms
        self.p999Ms = p999Ms
        self.histogramBuckets = histogramBuckets
    }

    public static let empty = LatencyDistribution(
        countSamples: 0,
        meanMs: 0,
        stdDevMs: 0,
        minMs: 0,
        maxMs: 0,
        p50Ms: 0,
        p90Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        p999Ms: 0,
        histogramBuckets: []
    )

    public static func compute(fromLatenciesMs latencies: [Double], bucketCount: Int = 32) -> LatencyDistribution {
        guard !latencies.isEmpty else {
            return .empty
        }

        let sorted = latencies.sorted()
        let count = sorted.count

        let minValue = sorted.first ?? 0
        let maxValue = sorted.last ?? 0

        let meanValue = mean(in: sorted)
        let stdDevValue = standardDeviation(in: sorted, mean: meanValue)

        let buckets = buildHistogramBuckets(
            sorted: sorted,
            bucketCount: max(1, bucketCount),
            minValue: minValue,
            maxValue: maxValue
        )

        return LatencyDistribution(
            countSamples: count,
            meanMs: meanValue,
            stdDevMs: stdDevValue,
            minMs: minValue,
            maxMs: maxValue,
            p50Ms: percentile(0.50, inSorted: sorted),
            p90Ms: percentile(0.90, inSorted: sorted),
            p95Ms: percentile(0.95, inSorted: sorted),
            p99Ms: percentile(0.99, inSorted: sorted),
            p999Ms: percentile(0.999, inSorted: sorted),
            histogramBuckets: buckets
        )
    }

    public func renderASCIIHistogram(width: Int = 40) -> String {
        guard !histogramBuckets.isEmpty, countSamples > 0 else {
            return "(no samples)"
        }

        let maxCount = histogramBuckets.reduce(0) { Swift.max($0, $1.count) }
        guard maxCount > 0 else {
            return "(empty histogram)"
        }

        let usableWidth = max(1, width)
        var lines: [String] = []
        let labelWidth = 18

        for bucket in histogramBuckets {
            let label = String(format: "%7.3f-%7.3f", bucket.lowerMs, bucket.upperMs)
            let paddedLabel = Self.padRightLatency(label, to: labelWidth)
            let barLength = Int((Double(bucket.count) / Double(maxCount)) * Double(usableWidth))
            let bar = String(repeating: "#", count: barLength)
            let countText = " (\(bucket.count))"
            lines.append("\(paddedLabel) | \(bar)\(countText)")
        }

        return lines.joined(separator: "\n")
    }

    public func cdfCSV() -> String {
        guard !histogramBuckets.isEmpty, countSamples > 0 else {
            return "latencyMs,cumulativeFraction\n"
        }

        var lines = ["latencyMs,cumulativeFraction"]
        var cumulative = 0
        let total = Double(countSamples)
        for bucket in histogramBuckets {
            cumulative += bucket.count
            let fraction = total > 0 ? Double(cumulative) / total : 0
            lines.append("\(Self.formatDouble(bucket.upperMs)),\(Self.formatDouble(fraction))")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    public func histogramCSV() -> String {
        var lines = ["lowerMs,upperMs,count"]
        for bucket in histogramBuckets {
            lines.append("\(Self.formatDouble(bucket.lowerMs)),\(Self.formatDouble(bucket.upperMs)),\(bucket.count)")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func percentile(_ p: Double, inSorted sorted: [Double]) -> Double {
        guard !sorted.isEmpty else {
            return 0
        }
        let rank = Int(ceil(p * Double(sorted.count))) - 1
        let index = min(max(rank, 0), sorted.count - 1)
        return sorted[index]
    }

    private static func mean(in values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func standardDeviation(in values: [Double], mean meanValue: Double) -> Double {
        guard values.count > 1 else {
            return 0
        }
        let squaredSum = values.reduce(0.0) { running, value in
            let diff = value - meanValue
            return running + (diff * diff)
        }
        return sqrt(squaredSum / Double(values.count))
    }

    private static func buildHistogramBuckets(
        sorted: [Double],
        bucketCount: Int,
        minValue: Double,
        maxValue: Double
    ) -> [HistogramBucket] {
        if minValue == maxValue {
            return [HistogramBucket(lowerMs: minValue, upperMs: maxValue, count: sorted.count)]
        }

        let edges = makeBucketEdges(min: minValue, max: maxValue, bucketCount: bucketCount)
        guard edges.count >= 2 else {
            return [HistogramBucket(lowerMs: minValue, upperMs: maxValue, count: sorted.count)]
        }

        var counts = [Int](repeating: 0, count: edges.count - 1)
        for value in sorted {
            var placed = false
            for i in 0..<(edges.count - 1) {
                let upper = edges[i + 1]
                let isLast = i == edges.count - 2
                if value < upper || (isLast && value <= upper) {
                    counts[i] += 1
                    placed = true
                    break
                }
            }
            if !placed {
                counts[counts.count - 1] += 1
            }
        }

        var buckets: [HistogramBucket] = []
        buckets.reserveCapacity(counts.count)
        for i in 0..<counts.count {
            buckets.append(
                HistogramBucket(lowerMs: edges[i], upperMs: edges[i + 1], count: counts[i])
            )
        }
        return buckets
    }

    private static func makeBucketEdges(min minValue: Double, max maxValue: Double, bucketCount: Int) -> [Double] {
        let safeMin = minValue > 0 ? minValue : 1e-6
        let safeMax = maxValue > safeMin ? maxValue : safeMin * 10
        let canLog = minValue > 0
        if canLog {
            let logMin = log10(safeMin)
            let logMax = log10(safeMax)
            let step = (logMax - logMin) / Double(bucketCount)
            var edges: [Double] = []
            edges.reserveCapacity(bucketCount + 1)
            for i in 0...bucketCount {
                edges.append(pow(10.0, logMin + step * Double(i)))
            }
            edges[0] = minValue
            edges[edges.count - 1] = maxValue
            return edges
        }

        let step = (maxValue - minValue) / Double(bucketCount)
        var edges: [Double] = []
        edges.reserveCapacity(bucketCount + 1)
        for i in 0...bucketCount {
            edges.append(minValue + step * Double(i))
        }
        edges[edges.count - 1] = maxValue
        return edges
    }

    private static func padRightLatency(_ text: String, to width: Int) -> String {
        if text.count >= width {
            return text
        }
        return text + String(repeating: " ", count: width - text.count)
    }

    private static func formatDouble(_ value: Double) -> String {
        String(format: "%.6f", value)
    }
}
