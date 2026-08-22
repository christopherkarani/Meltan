import Foundation

extension Fixtures {
    /// Fraction of the exact top-k set recovered by the approximate set;
    /// 0 when the exact set is empty.
    public static func recall(approx: Set<UInt32>, exact: Set<UInt32>) -> Double {
        guard !exact.isEmpty else {
            return 0
        }
        return Double(approx.intersection(exact).count) / Double(exact.count)
    }
}
