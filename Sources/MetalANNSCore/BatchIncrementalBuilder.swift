import Foundation

public enum BatchIncrementalBuilder {
    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    /// Insert multiple vectors at once using:
    /// 1) sequential forward neighbor assignment and
    /// 2) one deferred batched reverse-update pass.
    public static func batchInsert(
        vectors newVectors: [[Float]],
        startingAt startSlot: Int,
        into graph: GraphBuffer,
        vectorStorage: any VectorStorage,
        entryPoint: UInt32,
        metric: Metric,
        degree: Int
    ) throws {
        guard !newVectors.isEmpty else {
            return
        }
        guard startSlot >= 0 else {
            throw ANNSError.constructionFailed("Start slot must be non-negative")
        }
        guard degree > 0 else {
            throw ANNSError.constructionFailed("Degree must be greater than 0")
        }
        guard startSlot + newVectors.count <= graph.capacity,
              startSlot + newVectors.count <= vectorStorage.capacity else {
            throw ANNSError.constructionFailed("Batch insert exceeds capacity")
        }

        let existingNodeCount = graph.nodeCount
        guard existingNodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot batch insert into empty graph")
        }
        guard entryPoint < graph.capacity, Int(entryPoint) < max(1, existingNodeCount) else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        for (offset, vector) in newVectors.enumerated() {
            guard vector.count == vectorStorage.dim else {
                throw ANNSError.dimensionMismatch(expected: vectorStorage.dim, got: vector.count)
            }
            let internalID = startSlot + offset
            try IncrementalBuilder.insert(
                vector: vector,
                at: internalID,
                into: graph,
                vectors: vectorStorage,
                entryPoint: entryPoint,
                metric: metric,
                degree: degree
            )
        }

        if graph.nodeCount < startSlot + newVectors.count {
            graph.setCount(startSlot + newVectors.count)
        }
    }


}
