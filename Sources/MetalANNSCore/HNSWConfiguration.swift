import Foundation

public struct HNSWConfiguration: Sendable, Codable, Equatable {
    /// Whether HNSW layers are enabled for CPU search.
    public var enabled: Bool

    /// Connection limit per skip layer.
    // swift-format-ignore: AlwaysUseLowerCamelCase
    public var M: Int

    /// Maximum skip layer count.
    public var maxLayers: Int

    public static let `default` = HNSWConfiguration(
        enabled: true,
        M: 8,
        maxLayers: 6
    )

    public init(
        enabled: Bool = true,
        M: Int = 8,
        maxLayers: Int = 6
    ) {
        self.enabled = enabled
        self.M = max(1, M)
        self.maxLayers = max(0, maxLayers)
    }
}
