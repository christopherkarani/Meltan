import Foundation

public struct RepairConfiguration: Sendable, Codable, Equatable {
    /// Trigger repair every N inserts. Set to 0 to disable automatic repair.
    public var repairInterval: Int

    /// Number of hops to expand from each recent node to collect the repair neighborhood.
    /// 1 = direct neighbors only. 2 = neighbors of neighbors.
    public var repairDepth: Int

    /// Number of localized NN-Descent iterations per repair cycle.
    public var repairIterations: Int

    /// Whether automatic repair is enabled.
    public var enabled: Bool

    public static let `default` = RepairConfiguration(
        repairInterval: 100,
        repairDepth: 2,
        repairIterations: 5,
        enabled: true
    )

    public init(
        repairInterval: Int = 100,
        repairDepth: Int = 2,
        repairIterations: Int = 5,
        enabled: Bool = true
    ) {
        self.repairInterval = max(0, repairInterval)
        self.repairDepth = max(1, min(repairDepth, 3))
        self.repairIterations = max(1, min(repairIterations, 20))
        self.enabled = enabled
    }
}
