import Foundation
import Metal
import MetalANNSCore

internal struct _PersistedMetadata: Codable, Sendable {
    let configuration: IndexConfiguration
    let softDeletion: SoftDeletion
    let metadataStore: MetadataStore?
    let idMap: IDMap?
}

public actor GraphIndex {
    internal static let fullGPUMaxEF = 256
    internal static let minGPUConstructionNodeCount = 4_096
    internal static let minHybridGPUSearchNodeCount = 4_096
    internal static let minHybridGPUSearchWork = 16_384

    internal var configuration: IndexConfiguration
    internal var context: MetalContext?
    internal var vectors: (any VectorStorage)?
    internal var graph: GraphBuffer?
    internal var idMap: IDMap
    internal var softDeletion: SoftDeletion
    internal var metadataStore: MetadataStore
    internal var entryPoint: UInt32
    internal var isBuilt: Bool
    internal var isReadOnlyLoadedIndex: Bool
    internal var mmapLifetime: AnyObject?
    internal var pendingRepairIDs: [UInt32] = []
    internal var hnsw: HNSWLayers?
    public var metrics: IndexMetrics? = nil

    public init(configuration: IndexConfiguration = .default, context: MetalContext? = nil) {
        self.configuration = configuration
        self.context = context ?? (try? MetalContext())
        self.vectors = nil
        self.graph = nil
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.entryPoint = 0
        self.isBuilt = false
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.hnsw = nil
    }

    init(configuration: IndexConfiguration, exactContext: MetalContext?) {
        self.configuration = configuration
        self.context = exactContext
        self.vectors = nil
        self.graph = nil
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.entryPoint = 0
        self.isBuilt = false
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.hnsw = nil
    }

}
