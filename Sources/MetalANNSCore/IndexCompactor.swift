import Foundation
import Metal

package enum IndexCompactor {
    package struct CompactionResult {
        package let vectors: any VectorStorage
        package let graph: GraphBuffer
        package let idMap: IDMap
        package let entryPoint: UInt32

        package init(
            vectors: any VectorStorage,
            graph: GraphBuffer,
            idMap: IDMap,
            entryPoint: UInt32
        ) {
            self.vectors = vectors
            self.graph = graph
            self.idMap = idMap
            self.entryPoint = entryPoint
        }
    }

    package static func compact(
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        softDeletion: SoftDeletion,
        metric: Metric,
        degree: Int,
        context: MetalContext?,
        maxIterations: Int,
        convergenceThreshold: Float,
        useFloat16: Bool,
        useBinary: Bool
    ) async throws -> CompactionResult {
        _ = graph

        let deletedIDs = softDeletion.allDeletedIDs
        var survivingOldIDs: [UInt32] = []
        survivingOldIDs.reserveCapacity(max(0, vectors.count - deletedIDs.count))

        for oldIndex in 0..<vectors.count {
            let oldID = UInt32(oldIndex)
            if !deletedIDs.contains(oldID) {
                survivingOldIDs.append(oldID)
            }
        }

        guard !survivingOldIDs.isEmpty else {
            throw ANNSError.constructionFailed("Cannot compact index with all nodes deleted")
        }

        let survivingCount = survivingOldIDs.count
        let newCapacity = max(2, survivingCount * 2)
        let targetDegree = max(1, degree)
        let device = context?.device

        let compactedVectors: any VectorStorage
        if useBinary {
            compactedVectors = try BinaryVectorBuffer(capacity: newCapacity, dim: vectors.dim, device: device)
        } else if useFloat16 {
            compactedVectors = try Float16VectorBuffer(capacity: newCapacity, dim: vectors.dim, device: device)
        } else {
            compactedVectors = try VectorBuffer(capacity: newCapacity, dim: vectors.dim, device: device)
        }

        var denseVectors: [[Float]] = []
        denseVectors.reserveCapacity(survivingCount)

        var rebuiltIDMap = IDMap()
        for (newIndex, oldID) in survivingOldIDs.enumerated() {
            let vector = vectors.vector(at: Int(oldID))
            denseVectors.append(vector)
            let storageVector: [Float]
            if useBinary {
                storageVector = vector.map { $0 > 0.5 ? 1.0 : -1.0 }
            } else {
                storageVector = vector
            }
            try compactedVectors.insert(vector: storageVector, at: newIndex)

            if let externalID = idMap.externalID(for: oldID) {
                guard rebuiltIDMap.assign(externalID: externalID) != nil else {
                    throw ANNSError.idAlreadyExists(externalID)
                }
            } else if let numericID = idMap.numericID(for: oldID) {
                guard rebuiltIDMap.assign(numericID: numericID) != nil else {
                    throw ANNSError.idAlreadyExists(String(numericID))
                }
            } else {
                throw ANNSError.constructionFailed("Missing ID mapping for internal ID \(oldID)")
            }
        }
        compactedVectors.setCount(survivingCount)

        let compactedGraph = try GraphBuffer(capacity: newCapacity, degree: targetDegree, device: device)
        let rebuiltEntryPoint: UInt32

        if survivingCount == 1 {
            let ids = Array(repeating: UInt32.max, count: targetDegree)
            let distances = Array(repeating: Float.greatestFiniteMagnitude, count: targetDegree)
            try compactedGraph.setNeighbors(of: 0, ids: ids, distances: distances)
            compactedGraph.setCount(1)
            rebuiltEntryPoint = 0
        } else {
            let rebuildDegree = max(1, min(targetDegree, survivingCount - 1))
            if rebuildDegree == targetDegree {
                rebuiltEntryPoint = try await rebuildIntoGraph(
                    vectors: denseVectors,
                    vectorStorage: compactedVectors,
                    graph: compactedGraph,
                    metric: metric,
                    maxIterations: maxIterations,
                    convergenceThreshold: convergenceThreshold,
                    context: context
                )
            } else {
                let reducedGraph = try GraphBuffer(capacity: survivingCount, degree: rebuildDegree, device: device)
                let reducedEntryPoint = try await rebuildIntoGraph(
                    vectors: denseVectors,
                    vectorStorage: compactedVectors,
                    graph: reducedGraph,
                    metric: metric,
                    maxIterations: maxIterations,
                    convergenceThreshold: convergenceThreshold,
                    context: context
                )
                try expandGraph(from: reducedGraph, into: compactedGraph, nodeCount: survivingCount)
                compactedGraph.setCount(survivingCount)
                rebuiltEntryPoint = reducedEntryPoint
            }
        }

        // FIXME: Major — MetadataStore is not passed through compaction.
        // Any metadata associated with surviving nodes is permanently lost.
        return CompactionResult(
            vectors: compactedVectors,
            graph: compactedGraph,
            idMap: rebuiltIDMap,
            entryPoint: rebuiltEntryPoint
        )
    }

    private static func rebuildIntoGraph(
        vectors denseVectors: [[Float]],
        vectorStorage: any VectorStorage,
        graph: GraphBuffer,
        metric: Metric,
        maxIterations: Int,
        convergenceThreshold: Float,
        context: MetalContext?
    ) async throws -> UInt32 {
        let nodeCount = denseVectors.count
        let entryPoint: UInt32

        if let context, metric != .hamming {
            try await NNDescentGPU.build(
                context: context,
                vectors: vectorStorage,
                graph: graph,
                nodeCount: nodeCount,
                metric: metric,
                maxIterations: maxIterations,
                convergenceThreshold: convergenceThreshold
            )
            entryPoint = 0
        } else {
            let cpuResult = try await NNDescentCPU.build(
                vectors: denseVectors,
                degree: graph.degree,
                metric: metric,
                maxIterations: maxIterations,
                convergenceThreshold: convergenceThreshold
            )
            try populate(graph: graph, from: cpuResult.graph)
            graph.setCount(nodeCount)
            entryPoint = cpuResult.entryPoint
        }

        try GraphPruner.prune(
            graph: graph,
            vectors: vectorStorage,
            nodeCount: nodeCount,
            metric: metric
        )
        if graph.nodeCount < nodeCount {
            graph.setCount(nodeCount)
        }
        return entryPoint
    }

    private static func populate(graph: GraphBuffer, from graphData: [[(UInt32, Float)]]) throws {
        let degree = graph.degree
        for node in 0..<graphData.count {
            let neighbors = graphData[node]
            var ids = Array(repeating: UInt32.max, count: degree)
            var distances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)
            for slot in 0..<min(degree, neighbors.count) {
                ids[slot] = neighbors[slot].0
                distances[slot] = neighbors[slot].1
            }
            try graph.setNeighbors(of: node, ids: ids, distances: distances)
        }
    }

    private static func expandGraph(
        from reduced: GraphBuffer,
        into expanded: GraphBuffer,
        nodeCount: Int
    ) throws {
        for nodeID in 0..<nodeCount {
            let reducedIDs = reduced.neighborIDs(of: nodeID)
            let reducedDistances = reduced.neighborDistances(of: nodeID)

            var ids = Array(repeating: UInt32.max, count: expanded.degree)
            var distances = Array(repeating: Float.greatestFiniteMagnitude, count: expanded.degree)

            for slot in 0..<min(reduced.degree, expanded.degree) {
                ids[slot] = reducedIDs[slot]
                distances[slot] = reducedDistances[slot]
            }

            try expanded.setNeighbors(of: nodeID, ids: ids, distances: distances)
        }
    }
}
