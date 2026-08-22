import Foundation

public actor IndexMetrics {
    public private(set) var insertCount: Int = 0
    public private(set) var searchCount: Int = 0
    public private(set) var batchSearchCount: Int = 0
    public private(set) var mergeCount: Int = 0

    private static let bucketBoundsNs: [UInt64] = [
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
        200_000_000,
        500_000_000,
        1_000_000_000,
    ]

    public private(set) var searchLatencyHistogram: [Int] = Array(repeating: 0, count: 10)
    public private(set) var insertLatencyHistogram: [Int] = Array(repeating: 0, count: 10)

    public init() {}

    func recordInsert(durationNs: UInt64) {
        insertCount += 1
        insertLatencyHistogram[bucketIndex(for: durationNs)] += 1
    }

    func recordBatchInsert(count: Int, durationNs: UInt64) {
        guard count > 0 else {
            return
        }
        insertCount += count
        insertLatencyHistogram[bucketIndex(for: durationNs)] += 1
    }

    func recordSearch(durationNs: UInt64) {
        searchCount += 1
        searchLatencyHistogram[bucketIndex(for: durationNs)] += 1
    }

    func recordBatchSearch() {
        batchSearchCount += 1
    }

    func recordMerge() {
        mergeCount += 1
    }

    public func snapshot() -> MetricsSnapshot {
        MetricsSnapshot(
            insertCount: insertCount,
            searchCount: searchCount,
            batchSearchCount: batchSearchCount,
            mergeCount: mergeCount,
            searchP50LatencyMs: percentile(0.50, from: searchLatencyHistogram),
            searchP99LatencyMs: percentile(0.99, from: searchLatencyHistogram),
            timestamp: Date()
        )
    }

    private func bucketIndex(for ns: UInt64) -> Int {
        for (index, bound) in Self.bucketBoundsNs.enumerated() where ns < bound {
            return index
        }
        return Self.bucketBoundsNs.count - 1
    }

    private func percentile(_ p: Double, from histogram: [Int]) -> Double {
        let total = histogram.reduce(0, +)
        guard total > 0 else {
            return 0.0
        }

        let target = max(1, Int(ceil(Double(total) * p)))
        var cumulative = 0

        for (index, count) in histogram.enumerated() {
            cumulative += count
            if cumulative >= target {
                let lowerNs = index == 0 ? 0.0 : Double(Self.bucketBoundsNs[index - 1])
                let upperNs = Double(Self.bucketBoundsNs[index])
                return (lowerNs + upperNs) / 2.0 / 1_000_000.0
            }
        }

        let last = Self.bucketBoundsNs.count - 1
        return Double(Self.bucketBoundsNs[last]) / 1_000_000.0
    }
}
