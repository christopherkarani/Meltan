import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "HNSWBuilder")

public enum HNSWBuilder {
    private static let defaultML = 1.4426950408889634

    /// Build HNSW skip layers from a complete base graph.
    public static func buildLayers(
        vectors: any VectorStorage,
        graph: [[(UInt32, Float)]],
        nodeCount: Int,
        metric: Metric,
        config: HNSWConfiguration = .default
    ) throws(ANNSError) -> HNSWLayers {
        guard config.enabled else {
            return HNSWLayers()
        }
        guard nodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot build HNSW layers with empty graph")
        }
        guard nodeCount == graph.count else {
            throw ANNSError.constructionFailed("Graph size does not match node count")
        }
        guard vectors.count >= nodeCount else {
            throw ANNSError.constructionFailed("Vector storage count is less than node count")
        }

        var nodeLevels = Array(repeating: 0, count: nodeCount)
        var maxLayerAssigned = 0

        for nodeID in 0..<nodeCount {
            let level = assignLevel(mL: defaultML, maxLayers: config.maxLayers)
            nodeLevels[nodeID] = level
            maxLayerAssigned = max(maxLayerAssigned, level)
        }

        var entryPoint: UInt32 = 0
        var entryLevel = nodeLevels[0]
        for nodeID in 1..<nodeCount {
            if nodeLevels[nodeID] > entryLevel {
                entryLevel = nodeLevels[nodeID]
                entryPoint = UInt32(nodeID)
            }
        }

        logger.debug("Assigned HNSW levels: maxLayer=\(maxLayerAssigned), entryPoint=\(entryPoint)")

        var skipLayers: [SkipLayer] = []
        if maxLayerAssigned > 0 {
            for layer in 1...maxLayerAssigned {
                let skipLayer = buildSkipLayer(
                    at: layer,
                    nodeLevels: nodeLevels,
                    vectors: vectors,
                    graph: graph,
                    nodeCount: nodeCount,
                    metric: metric,
                    M: config.M
                )
                skipLayers.append(skipLayer)
            }
        }

        return HNSWLayers(
            layers: skipLayers,
            maxLayer: maxLayerAssigned,
            mL: defaultML,
            entryPoint: entryPoint
        )
    }

    /// Assign level via exponential distribution:
    /// floor(-ln(uniform(0, 1)) * mL), capped by maxLayers.
    static func assignLevel(mL: Double, maxLayers: Int) -> Int {
        guard maxLayers > 0 else {
            return 0
        }
        let uniform = Double.random(in: Double.ulpOfOne..<1.0)
        let raw = Int(floor(-log(uniform) * mL))
        return min(max(0, raw), maxLayers)
    }

    // swift-format-ignore: AlwaysUseLowerCamelCase
    private static func buildSkipLayer(
        at layer: Int,
        nodeLevels: [Int],
        vectors: any VectorStorage,
        graph: [[(UInt32, Float)]],
        nodeCount: Int,
        metric: Metric,
        M: Int
    ) -> SkipLayer {
        var nodesAtLayer: [UInt32] = []
        nodesAtLayer.reserveCapacity(nodeCount / 2)
        for nodeID in 0..<nodeCount where nodeLevels[nodeID] >= layer {
            nodesAtLayer.append(UInt32(nodeID))
        }

        guard !nodesAtLayer.isEmpty else {
            return SkipLayer()
        }

        var nodeToLayerIndex: [UInt32: UInt32] = [:]
        nodeToLayerIndex.reserveCapacity(nodesAtLayer.count)
        var layerIndexToNode: [UInt32] = []
        layerIndexToNode.reserveCapacity(nodesAtLayer.count)

        for (index, nodeID) in nodesAtLayer.enumerated() {
            let localIndex = UInt32(index)
            nodeToLayerIndex[nodeID] = localIndex
            layerIndexToNode.append(nodeID)
        }

        var adjacency = Array(repeating: [UInt32](), count: nodesAtLayer.count)
        let layerNodeSet = Set(nodesAtLayer)
        let targetCandidatePool = max(M * 8, M)
        let cachedVectors = nodesAtLayer.map { vectors.vector(at: Int($0)) }
        let layerIndexByNode = Dictionary(
            uniqueKeysWithValues: nodesAtLayer.enumerated().map { (offset, nodeID) in
                (nodeID, offset)
            })

        for (layerIndex, nodeID) in nodesAtLayer.enumerated() {
            let sourceVector = cachedVectors[layerIndex]
            let candidateIDs = collectCandidates(
                for: nodeID,
                graph: graph,
                nodesAtLayer: nodesAtLayer,
                layerNodeSet: layerNodeSet,
                targetCount: targetCandidatePool
            )
            var bestNeighbors = BinaryHeap<(UInt32, Float)> { lhs, rhs in
                lhs.1 > rhs.1
            }
            for otherID in candidateIDs where otherID != nodeID {
                let otherVector: [Float]
                if let cachedIndex = layerIndexByNode[otherID] {
                    otherVector = cachedVectors[cachedIndex]
                } else {
                    otherVector = vectors.vector(at: Int(otherID))
                }
                let distance = SIMDDistance.distance(sourceVector, otherVector, metric: metric)
                if bestNeighbors.count < max(1, M) {
                    bestNeighbors.push((otherID, distance))
                } else if let worst = bestNeighbors.peek, distance < worst.1 {
                    bestNeighbors.replaceTop(with: (otherID, distance))
                }
            }

            adjacency[layerIndex] = bestNeighbors.unorderedElements()
                .sorted { $0.1 < $1.1 }
                .map { $0.0 }
        }

        return SkipLayer(
            nodeToLayerIndex: nodeToLayerIndex,
            layerIndexToNode: layerIndexToNode,
            adjacency: adjacency
        )
    }

    private static func collectCandidates(
        for nodeID: UInt32,
        graph: [[(UInt32, Float)]],
        nodesAtLayer: [UInt32],
        layerNodeSet: Set<UInt32>,
        targetCount: Int
    ) -> [UInt32] {
        guard !nodesAtLayer.isEmpty, targetCount > 0 else {
            return []
        }

        var candidates = Set<UInt32>()
        var visited = Set<UInt32>()
        visited.insert(nodeID)

        var frontier: [UInt32] = []
        if Int(nodeID) < graph.count {
            for (neighbor, _) in graph[Int(nodeID)] where neighbor != UInt32.max {
                if neighbor != nodeID, layerNodeSet.contains(neighbor), visited.insert(neighbor).inserted {
                    candidates.insert(neighbor)
                    frontier.append(neighbor)
                }
            }
        }

        let maxExpandedNodes = max(targetCount * 4, targetCount + 8)
        var frontierIndex = 0
        while candidates.count < targetCount && frontierIndex < frontier.count && visited.count < maxExpandedNodes {
            let current = frontier[frontierIndex]
            frontierIndex += 1
            guard Int(current) < graph.count else {
                continue
            }

            for (neighbor, _) in graph[Int(current)] where neighbor != UInt32.max {
                if neighbor == nodeID || !layerNodeSet.contains(neighbor) {
                    continue
                }
                if visited.insert(neighbor).inserted {
                    candidates.insert(neighbor)
                    frontier.append(neighbor)
                    if candidates.count >= targetCount || visited.count >= maxExpandedNodes {
                        break
                    }
                }
            }
        }

        if candidates.count < targetCount {
            // Deterministic supplement to maintain connectivity without O(N^2) all-pairs scans.
            let stride = max(1, nodesAtLayer.count / max(targetCount, 1))
            var probe = Int((UInt64(nodeID) * 2_654_435_761) % UInt64(nodesAtLayer.count))
            var scanned = 0

            while candidates.count < targetCount && scanned < nodesAtLayer.count {
                let candidate = nodesAtLayer[probe]
                if candidate != nodeID {
                    candidates.insert(candidate)
                }
                probe = (probe + stride) % nodesAtLayer.count
                scanned += 1
            }
        }

        return Array(candidates)
    }
}
