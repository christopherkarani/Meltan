import MetalANNSCore

/// Controls how _StreamingIndex merges its delta into the base index.
public struct StreamingConfiguration: Sendable, Codable, Equatable {
    /// Maximum number of vectors in the delta index before a merge is triggered.
    public var deltaCapacity: Int

    /// Strategy for merging delta into the base index.
    public var mergeStrategy: MergeStrategy

    /// Configuration used for both base and delta GraphIndex instances.
    public var indexConfiguration: IndexConfiguration

    public static let `default` = StreamingConfiguration(
        deltaCapacity: 10_000,
        mergeStrategy: .background,
        indexConfiguration: .default
    )

    public init(
        deltaCapacity: Int = 10_000,
        mergeStrategy: MergeStrategy = .background,
        indexConfiguration: IndexConfiguration = .default
    ) {
        self.deltaCapacity = max(1, deltaCapacity)
        self.mergeStrategy = mergeStrategy
        self.indexConfiguration = indexConfiguration
    }

    public static func == (lhs: StreamingConfiguration, rhs: StreamingConfiguration) -> Bool {
        lhs.deltaCapacity == rhs.deltaCapacity
            && lhs.mergeStrategy == rhs.mergeStrategy
            && lhs.indexConfiguration.degree == rhs.indexConfiguration.degree
            && lhs.indexConfiguration.metric.rawValue == rhs.indexConfiguration.metric.rawValue
            && lhs.indexConfiguration.efConstruction == rhs.indexConfiguration.efConstruction
            && lhs.indexConfiguration.efSearch == rhs.indexConfiguration.efSearch
            && lhs.indexConfiguration.maxIterations == rhs.indexConfiguration.maxIterations
            && lhs.indexConfiguration.useFloat16 == rhs.indexConfiguration.useFloat16
            && lhs.indexConfiguration.useBinary == rhs.indexConfiguration.useBinary
            && lhs.indexConfiguration.convergenceThreshold == rhs.indexConfiguration.convergenceThreshold
            && lhs.indexConfiguration.hnswConfiguration.enabled == rhs.indexConfiguration.hnswConfiguration.enabled
            && lhs.indexConfiguration.hnswConfiguration.M == rhs.indexConfiguration.hnswConfiguration.M
            && lhs.indexConfiguration.hnswConfiguration.maxLayers == rhs.indexConfiguration.hnswConfiguration.maxLayers
            && lhs.indexConfiguration.repairConfiguration.repairInterval == rhs.indexConfiguration.repairConfiguration.repairInterval
            && lhs.indexConfiguration.repairConfiguration.repairDepth == rhs.indexConfiguration.repairConfiguration.repairDepth
            && lhs.indexConfiguration.repairConfiguration.repairIterations == rhs.indexConfiguration.repairConfiguration.repairIterations
            && lhs.indexConfiguration.repairConfiguration.enabled == rhs.indexConfiguration.repairConfiguration.enabled
    }

    /// Determines when and how merges happen.
    public enum MergeStrategy: Sendable, Codable, Equatable {
        /// Merge runs as a detached background task.
        case background

        /// Merge runs inline; inserts block until merge completes.
        case blocking
    }
}
