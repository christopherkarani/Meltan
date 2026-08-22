import Foundation

public enum GraphPruner {
    /// Prune redundant edges from the graph using path-based diversification.
    /// For each node, removes neighbors that are reachable via shorter paths
    /// through other neighbors.
    ///
    /// FIXME: Major — candidates are iterated in adjacency-slot order, not sorted
    /// by distance. Since pruning is order-dependent, unsorted input may keep a
    /// farther neighbor and prune a closer one. Sort candidates by distance first.
    public static func prune(
        graph: GraphBuffer,
        vectors: any VectorStorage,
        nodeCount: Int,
        metric: Metric
    ) throws {
        guard nodeCount >= 0, nodeCount <= graph.nodeCount, nodeCount <= vectors.count else {
            throw ANNSError.constructionFailed("nodeCount is out of bounds for graph/vector counts")
        }

        let degree = graph.degree
        guard degree > 0 else {
            return
        }

        if let floatVectors = vectors as? VectorBuffer {
            try pruneFloat32(
                graph: graph,
                vectors: floatVectors,
                nodeCount: nodeCount,
                metric: metric
            )
            return
        }

        for nodeID in 0..<nodeCount {
            let neighborIDs = graph.neighborIDs(of: nodeID)
            let neighborDistances = graph.neighborDistances(of: nodeID)

            var candidates: [(id: UInt32, distance: Float, vector: [Float])] = []
            candidates.reserveCapacity(degree)
            for slot in 0..<degree {
                let neighborID = neighborIDs[slot]
                if neighborID == UInt32.max {
                    continue
                }
                let neighborIndex = Int(neighborID)
                if neighborIndex < 0 || neighborIndex >= nodeCount {
                    continue
                }
                candidates.append((neighborID, neighborDistances[slot], vectors.vector(at: neighborIndex)))
            }

            var prunedIDs: [UInt32] = []
            var prunedDistances: [Float] = []
            var prunedVectors: [[Float]] = []
            prunedIDs.reserveCapacity(degree)
            prunedDistances.reserveCapacity(degree)
            prunedVectors.reserveCapacity(degree)

            for candidate in candidates {
                var isRedundant = false

                for selectedVector in prunedVectors {
                    let pathDistance = SIMDDistance.distance(
                        selectedVector,
                        candidate.vector,
                        metric: metric
                    )
                    if pathDistance < candidate.distance {
                        isRedundant = true
                        break
                    }
                }

                if !isRedundant {
                    prunedIDs.append(candidate.id)
                    prunedDistances.append(candidate.distance)
                    prunedVectors.append(candidate.vector)
                }
            }

            if prunedIDs.count < degree {
                prunedIDs.append(contentsOf: Array(repeating: UInt32.max, count: degree - prunedIDs.count))
                prunedDistances.append(
                    contentsOf: Array(
                        repeating: Float.greatestFiniteMagnitude,
                        count: degree - prunedDistances.count
                    ))
            } else if prunedIDs.count > degree {
                prunedIDs = Array(prunedIDs.prefix(degree))
                prunedDistances = Array(prunedDistances.prefix(degree))
            }

            try graph.setNeighbors(of: nodeID, ids: prunedIDs, distances: prunedDistances)
        }
    }

    private static func pruneFloat32(
        graph: GraphBuffer,
        vectors: VectorBuffer,
        nodeCount: Int,
        metric: Metric
    ) throws {
        let degree = graph.degree
        let dim = vectors.dim
        let rawVectors = vectors.floatPointer

        for nodeID in 0..<nodeCount {
            let neighborIDs = graph.neighborIDs(of: nodeID)
            let neighborDistances = graph.neighborDistances(of: nodeID)

            var candidates: [(id: UInt32, distance: Float)] = []
            candidates.reserveCapacity(degree)
            for slot in 0..<degree {
                let neighborID = neighborIDs[slot]
                if neighborID == UInt32.max {
                    continue
                }
                let neighborIndex = Int(neighborID)
                if neighborIndex < 0 || neighborIndex >= nodeCount {
                    continue
                }
                candidates.append((neighborID, neighborDistances[slot]))
            }

            var prunedIDs: [UInt32] = []
            var prunedDistances: [Float] = []
            prunedIDs.reserveCapacity(degree)
            prunedDistances.reserveCapacity(degree)

            for candidate in candidates {
                let candidateBase = rawVectors.baseAddress!.advanced(by: Int(candidate.id) * dim)
                var isRedundant = false

                for selectedID in prunedIDs {
                    let selectedBase = rawVectors.baseAddress!.advanced(by: Int(selectedID) * dim)
                    let pathDistance = SIMDDistance.distance(
                        selectedBase,
                        candidateBase,
                        dim: dim,
                        metric: metric
                    )
                    if pathDistance < candidate.distance {
                        isRedundant = true
                        break
                    }
                }

                if !isRedundant {
                    prunedIDs.append(candidate.id)
                    prunedDistances.append(candidate.distance)
                }
            }

            if prunedIDs.count < degree {
                prunedIDs.append(contentsOf: Array(repeating: UInt32.max, count: degree - prunedIDs.count))
                prunedDistances.append(
                    contentsOf: Array(
                        repeating: Float.greatestFiniteMagnitude,
                        count: degree - prunedDistances.count
                    ))
            } else if prunedIDs.count > degree {
                prunedIDs = Array(prunedIDs.prefix(degree))
                prunedDistances = Array(prunedDistances.prefix(degree))
            }

            try graph.setNeighbors(of: nodeID, ids: prunedIDs, distances: prunedDistances)
        }
    }
}
