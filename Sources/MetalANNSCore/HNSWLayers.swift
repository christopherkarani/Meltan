import Foundation

/// Skip layer in the HNSW hierarchy.
package struct SkipLayer: Sendable, Codable {
    /// Maps graph node ID -> layer-local node index.
    package var nodeToLayerIndex: [UInt32: UInt32]

    /// Maps layer-local index -> graph node ID.
    package var layerIndexToNode: [UInt32]

    /// Layer adjacency lists indexed by layer-local index.
    package var adjacency: [[UInt32]]

    package init(
        nodeToLayerIndex: [UInt32: UInt32] = [:],
        layerIndexToNode: [UInt32] = [],
        adjacency: [[UInt32]] = []
    ) {
        self.nodeToLayerIndex = nodeToLayerIndex
        self.layerIndexToNode = layerIndexToNode
        self.adjacency = adjacency
    }
}

/// Complete HNSW skip-layer structure. Layer 0 uses the base graph.
package struct HNSWLayers: Sendable {
    /// Skip layers where `layers[0]` corresponds to layer 1.
    package let layers: [SkipLayer]

    /// Maximum assigned layer in the hierarchy.
    package let maxLayer: Int

    /// Level multiplier (1 / ln(2)).
    package let mL: Double

    /// Entry point in the highest populated layer.
    package let entryPoint: UInt32

    package init(
        layers: [SkipLayer] = [],
        maxLayer: Int = 0,
        mL: Double = 1.4426950408889634,
        entryPoint: UInt32 = 0
    ) {
        self.layers = layers
        self.maxLayer = maxLayer
        self.mL = mL
        self.entryPoint = entryPoint
    }

    /// Returns neighbor node IDs for `nodeID` at a skip layer.
    package func neighbors(of nodeID: UInt32, at layer: Int) -> [UInt32]? {
        guard layer > 0, layer <= maxLayer else {
            return nil
        }
        let skipLayer = layers[layer - 1]
        guard let layerIndex = skipLayer.nodeToLayerIndex[nodeID] else {
            return nil
        }
        return skipLayer.adjacency[Int(layerIndex)]
    }
}
