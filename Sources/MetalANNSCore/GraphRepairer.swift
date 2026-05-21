import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "GraphRepairer")

public enum GraphRepairer {
    @discardableResult
    public static func repair(
        recentIDs: [UInt32],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        config: RepairConfiguration,
        metric: Metric
    ) throws(ANNSError) -> Int {
        guard config.enabled else {
            return 0
        }

        guard !recentIDs.isEmpty else {
            return 0
        }

        let sanitizedIDs = recentIDs.filter {
            $0 != UInt32.max && Int($0) < graph.nodeCount
        }
        guard !sanitizedIDs.isEmpty else {
            return 0
        }

        let neighborhoods = collectNeighborhood(
            seeds: sanitizedIDs,
            graph: graph,
            depth: config.repairDepth
        )

        guard !neighborhoods.isEmpty else {
            return 0
        }

        let repairNodes = neighborhoods.sorted()
        let baselineDiversity = averageFarthestFiniteDistance(nodeIDs: repairNodes, graph: graph)

        var snapshot: [(node: Int, ids: [UInt32], distances: [Float])] = []
        snapshot.reserveCapacity(repairNodes.count)
        for nodeID in repairNodes {
            let node = Int(nodeID)
            snapshot.append((
                node: node,
                ids: graph.neighborIDs(of: node),
                distances: graph.neighborDistances(of: node)
            ))
        }

        let updates = try localNNDescent(
            nodes: neighborhoods,
            focusNodes: Set(sanitizedIDs),
            vectors: vectors,
            graph: graph,
            metric: metric,
            iterations: config.repairIterations
        )
        guard updates > 0 else {
            return 0
        }

        if baselineDiversity > 0 {
            let repairedDiversity = averageFarthestFiniteDistance(nodeIDs: repairNodes, graph: graph)
            if repairedDiversity < baselineDiversity * 0.98 {
                for entry in snapshot {
                    do {
                        try graph.setNeighbors(
                            of: entry.node,
                            ids: entry.ids,
                            distances: entry.distances
                        )
                    } catch {
                        throw ANNSError.constructionFailed("Failed to restore graph snapshot: \(error)")
                    }
                }
                logger.debug(
                    "GraphRepairer reverted rewiring due to diversity drop (\(baselineDiversity) -> \(repairedDiversity))"
                )
            }
        }

        // FIXME: Major — return value lies after rollback.
        // If diversity drops and we revert, we still return 'updates' (pre-reversion count).
        // Should return 0 when all changes were rolled back.
        return updates
    }

    private static func collectNeighborhood(seeds: [UInt32], graph: GraphBuffer, depth: Int) -> Set<UInt32> {
        guard graph.nodeCount > 0 else {
            return []
        }

        var visited = Set<UInt32>()
        var frontier = Set<UInt32>()

        let normalizedDepth = max(1, min(depth, 3))

        for seed in seeds {
            let seedIndex = Int(seed)
            if seed != UInt32.max && seedIndex >= 0 && seedIndex < graph.nodeCount {
                visited.insert(seed)
                frontier.insert(seed)
            }
        }

        for _ in 0..<normalizedDepth {
            var nextFrontier = Set<UInt32>()
            for node in frontier {
                let nodeIndex = Int(node)
                guard nodeIndex >= 0 && nodeIndex < graph.nodeCount else {
                    continue
                }

                for neighborID in graph.neighborIDs(of: nodeIndex) {
                    let neighborIndex = Int(neighborID)
                    if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= graph.nodeCount {
                        continue
                    }

                    if visited.insert(neighborID).inserted {
                        nextFrontier.insert(neighborID)
                    }
                }
            }
            if nextFrontier.isEmpty {
                break
            }
            frontier = nextFrontier
        }

        return visited
    }

    private static func localNNDescent(
        nodes: Set<UInt32>,
        focusNodes: Set<UInt32>,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        metric: Metric,
        iterations: Int
    ) throws(ANNSError) -> Int {
        guard !nodes.isEmpty else {
            return 0
        }

        let sortedIterations = max(1, iterations)
        let sortedNodes = nodes.sorted()
        let sortedFocusNodes = sortedNodes.filter { focusNodes.contains($0) }
        guard !sortedFocusNodes.isEmpty else {
            return 0
        }
        var totalUpdates = 0

        for iteration in 0..<sortedIterations {
            let maxUpdatesPerNode = 1
            var updatesPerNode: [UInt32: Int] = [:]
            var reverse: [UInt32: [UInt32]] = [:]
            for nodeID in sortedFocusNodes {
                let nodeIndex = Int(nodeID)
                guard nodeIndex >= 0 && nodeIndex < graph.nodeCount else {
                    continue
                }

                for neighborID in graph.neighborIDs(of: nodeIndex) {
                    let neighborIndex = Int(neighborID)
                    guard neighborID != UInt32.max && neighborIndex >= 0 && neighborIndex < graph.nodeCount else {
                        continue
                    }
                    if nodes.contains(neighborID) {
                        reverse[neighborID, default: []].append(nodeID)
                    }
                }
            }

            var iterationUpdates = 0

            for nodeID in sortedNodes {
                let nodeIndex = Int(nodeID)
                guard nodeIndex >= 0 && nodeIndex < graph.nodeCount else {
                    continue
                }

                var candidateSet = Set<UInt32>()
                let neighbors = graph.neighborIDs(of: nodeIndex)
                for neighborID in neighbors {
                    let neighborIndex = Int(neighborID)
                    guard neighborID != UInt32.max && neighborIndex >= 0 && neighborIndex < graph.nodeCount else {
                        continue
                    }
                    if nodes.contains(neighborID) {
                        candidateSet.insert(neighborID)
                    }
                }

                if let reverseNeighbors = reverse[nodeID] {
                    for reverseID in reverseNeighbors where nodes.contains(reverseID) {
                        candidateSet.insert(reverseID)
                    }
                }

                let candidateArray = candidateSet.sorted()
                guard candidateArray.count >= 2 else {
                    continue
                }

                for i in 0..<(candidateArray.count - 1) {
                    let aID = candidateArray[i]
                    let a = Int(aID)
                    for j in (i + 1)..<candidateArray.count {
                        let bID = candidateArray[j]
                        let b = Int(bID)
                        if a == b {
                            continue
                        }

                        let distance = SIMDDistance.distance(
                            vectors.vector(at: a),
                            vectors.vector(at: b),
                            metric: metric
                        )

                        if updatesPerNode[aID, default: 0] < maxUpdatesPerNode,
                           try tryImproveEdge(
                               node: a,
                               candidate: b,
                               distance: distance,
                               graph: graph
                           ) {
                            updatesPerNode[aID, default: 0] += 1
                            iterationUpdates += 1
                        }

                        if updatesPerNode[bID, default: 0] < maxUpdatesPerNode,
                           try tryImproveEdge(
                               node: b,
                               candidate: a,
                               distance: distance,
                               graph: graph
                           ) {
                            updatesPerNode[bID, default: 0] += 1
                            iterationUpdates += 1
                        }
                    }
                }
            }

            totalUpdates += iterationUpdates
            logger.debug("GraphRepairer iteration \(iteration): \(iterationUpdates) updates")

            let threshold = Float(graph.degree * nodes.count) * 0.001
            if Float(iterationUpdates) < threshold {
                logger.debug("GraphRepairer converged at iteration \(iteration + 1)")
                break
            }
        }

        return totalUpdates
    }

    private static func averageFarthestFiniteDistance(nodeIDs: [UInt32], graph: GraphBuffer) -> Float {
        guard !nodeIDs.isEmpty else {
            return 0
        }

        var total: Float = 0
        var contributingNodes = 0

        for nodeID in nodeIDs {
            let distances = graph.neighborDistances(of: Int(nodeID))
            var farthest: Float = -Float.greatestFiniteMagnitude
            for distance in distances where distance.isFinite && distance < Float.greatestFiniteMagnitude {
                if distance > farthest {
                    farthest = distance
                }
            }

            if farthest > -Float.greatestFiniteMagnitude {
                total += farthest
                contributingNodes += 1
            }
        }

        guard contributingNodes > 0 else {
            return 0
        }
        return total / Float(contributingNodes)
    }

    private static func tryImproveEdge(
        node: Int,
        candidate: Int,
        distance: Float,
        graph: GraphBuffer
    ) throws(ANNSError) -> Bool {
        guard node >= 0 && candidate >= 0 else {
            return false
        }
        guard node != candidate else {
            return false
        }

        let neighborIDs = graph.neighborIDs(of: node)
        if neighborIDs.contains(UInt32(candidate)) {
            return false
        }

        let neighborDistances = graph.neighborDistances(of: node)
        var updatedIDs = neighborIDs
        var updatedDistances = neighborDistances

        var worstIndex = 0
        var worstDistance = updatedDistances[0]
        if updatedDistances.count > 1 {
            for idx in 1..<updatedDistances.count {
                if updatedDistances[idx] > worstDistance {
                    worstDistance = updatedDistances[idx]
                    worstIndex = idx
                }
            }
        }

        if distance >= worstDistance {
            return false
        }

        updatedIDs[worstIndex] = UInt32(candidate)
        updatedDistances[worstIndex] = distance

        let sorted = zip(updatedIDs, updatedDistances).sorted { $0.1 < $1.1 }
        do {
            try graph.setNeighbors(
                of: node,
                ids: sorted.map(\.0),
                distances: sorted.map(\.1)
            )
        } catch {
            throw ANNSError.constructionFailed("Failed to update graph neighbors: \(error)")
        }

        return true
    }
}
