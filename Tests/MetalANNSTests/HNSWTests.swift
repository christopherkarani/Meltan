import Testing

@testable import MetalANNSCore

@Suite("HNSW Layer Tests")
struct HNSWTests {
    @Test("HNSWLayers stores neighbors")
    func hnswtStorageTest() {
        let layer = SkipLayer(
            nodeToLayerIndex: [0: 0, 1: 1],
            layerIndexToNode: [0, 1],
            adjacency: [[1], [0]]
        )
        let hnsw = HNSWLayers(layers: [layer], maxLayer: 1, entryPoint: 0)

        #expect(layer.nodeToLayerIndex[0] == 0)
        #expect(hnsw.neighbors(of: 0, at: 1) == [1])
    }
}
