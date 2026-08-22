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

    internal struct ReadyState {
        internal var vectors: any VectorStorage
        internal var graph: GraphBuffer
        internal var entryPoint: UInt32
    }

    internal struct LoadedState {
        internal let vectors: any VectorStorage
        internal let graph: GraphBuffer
        internal let entryPoint: UInt32
        internal let mmapLifetime: AnyObject
    }

    internal enum Lifecycle {
        case empty
        case ready(ReadyState)
        case loaded(LoadedState)

        internal var builtState: (vectors: any VectorStorage, graph: GraphBuffer, entryPoint: UInt32)? {
            switch self {
            case .empty:
                return nil
            case .ready(let state):
                return (state.vectors, state.graph, state.entryPoint)
            case .loaded(let state):
                return (state.vectors, state.graph, state.entryPoint)
            }
        }
    }

    internal var configuration: IndexConfiguration
    internal var context: MetalContext?
    internal var idMap: IDMap
    internal var softDeletion: SoftDeletion
    internal var metadataStore: MetadataStore
    internal var lifecycle: Lifecycle = .empty
    internal var pendingRepairIDs: [UInt32] = []
    internal var hnsw: HNSWLayers?
    public var metrics: IndexMetrics? = nil

    public init(configuration: IndexConfiguration = .default) {
        self.configuration = configuration
        self.context = try? MetalContext()
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.hnsw = nil
    }

    init(configuration: IndexConfiguration = .default, context: MetalContext?) {
        self.configuration = configuration
        self.context = context
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.hnsw = nil
    }

    internal func requireMutableState() throws(ANNSError) -> (
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: UInt32
    ) {
        switch lifecycle {
        case .empty:
            throw ANNSError.indexEmpty
        case .loaded:
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        case .ready(let state):
            return (state.vectors, state.graph, state.entryPoint)
        }
    }

}
