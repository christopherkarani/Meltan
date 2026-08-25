import Foundation
import Metal

package enum IncrementalBuilder {
    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    package static func insert(
        vector: [Float],
        at internalID: Int,
        into graph: GraphBuffer,
        vectors: any VectorStorage,
        entryPoint: UInt32,
        metric: Metric,
        degree: Int
    ) throws {
        guard internalID >= 0 && internalID < vectors.capacity else {
            throw ANNSError.constructionFailed("Internal ID out of bounds")
        }
        guard vector.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
        }
        guard degree > 0 else {
            throw ANNSError.constructionFailed("Degree must be greater than 0")
        }

        let existingNodeCount = graph.nodeCount
        guard entryPoint < graph.capacity && Int(entryPoint) < max(1, existingNodeCount) else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }
        guard existingNodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot incrementally insert into empty graph")
        }

        let nearest = nearestNeighbors(
            to: vector,
            graph: graph,
            vectors: vectors,
            entryPoint: Int(entryPoint),
            degree: degree,
            metric: metric
        )

        var neighborIDs = Array(repeating: UInt32.max, count: degree)
        var neighborDistances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)
        for index in 0..<min(degree, nearest.count) {
            neighborIDs[index] = nearest[index].nodeID
            neighborDistances[index] = nearest[index].distance
        }

        try graph.setNeighbors(of: internalID, ids: neighborIDs, distances: neighborDistances)

        var attached = false

        for neighborID in nearest {
            let neighborIndex = Int(neighborID.nodeID)
            guard neighborIndex >= 0 && neighborIndex < graph.nodeCount else {
                continue
            }

            let existingIDs = graph.neighborIDs(of: neighborIndex)
            let existingDistances = graph.neighborDistances(of: neighborIndex)

            if existingIDs.contains(UInt32(internalID)) {
                continue
            }

            let newDistance = SIMDDistance.distance(
                vector,
                vectors.vector(at: neighborIndex),
                metric: metric
            )

            if let replaceIndex = worstNeighborIndex(in: existingDistances),
                newDistance < existingDistances[replaceIndex]
            {
                var updatedIDs = existingIDs
                var updatedDistances = existingDistances
                updatedIDs[replaceIndex] = UInt32(internalID)
                updatedDistances[replaceIndex] = newDistance

                let sorted = zip(updatedIDs, updatedDistances).sorted { lhs, rhs in
                    lhs.1 < rhs.1
                }
                let sortedIDs = sorted.map(\.0)
                let sortedDistances = sorted.map(\.1)
                try graph.setNeighbors(of: neighborIndex, ids: sortedIDs, distances: sortedDistances)
                attached = true
            }
        }

        if !attached {
            let fallbackIndex = Int(entryPoint)
            guard fallbackIndex >= 0 && fallbackIndex < graph.nodeCount else {
                throw ANNSError.searchFailed("Fallback index out of bounds")
            }

            let existingIDs = graph.neighborIDs(of: fallbackIndex)
            if !existingIDs.contains(UInt32(internalID)) {
                let existingDistances = graph.neighborDistances(of: fallbackIndex)
                if let replaceIndex = worstNeighborIndex(in: existingDistances) {
                    let fallbackDistance = SIMDDistance.distance(
                        vector,
                        vectors.vector(at: fallbackIndex),
                        metric: metric
                    )

                    // FIXME: Major — should verify fallbackDistance < existingDistances[replaceIndex]
                    // before replacing. Unconditional replacement degrades graph quality over time.
                    var updatedIDs = existingIDs
                    var updatedDistances = existingDistances
                    updatedIDs[replaceIndex] = UInt32(internalID)
                    updatedDistances[replaceIndex] = fallbackDistance

                    let sorted = zip(updatedIDs, updatedDistances).sorted { lhs, rhs in
                        lhs.1 < rhs.1
                    }
                    let sortedIDs = sorted.map(\.0)
                    let sortedDistances = sorted.map(\.1)
                    try graph.setNeighbors(of: fallbackIndex, ids: sortedIDs, distances: sortedDistances)
                }
            }
        }

        if graph.nodeCount < internalID + 1 {
            graph.setCount(internalID + 1)
        }
    }

    private static func nearestNeighbors(
        to vector: [Float],
        graph: GraphBuffer,
        vectors: any VectorStorage,
        entryPoint: Int,
        degree: Int,
        metric: Metric
    ) -> [(nodeID: UInt32, distance: Float)] {
        let nodeCount = graph.nodeCount
        let ef = max(1, degree * 2)
        let efLimit = min(nodeCount, ef)

        let entryDistance = SIMDDistance.distance(
            vector,
            vectors.vector(at: entryPoint),
            metric: metric
        )
        var visited: Set<UInt32> = [UInt32(entryPoint)]
        var candidates: [Candidate] = [Candidate(nodeID: UInt32(entryPoint), distance: entryDistance)]
        var results: [Candidate] = [Candidate(nodeID: UInt32(entryPoint), distance: entryDistance)]

        while !candidates.isEmpty {
            let current = candidates.removeFirst()
            if results.count >= efLimit, let worst = results.last, current.distance > worst.distance {
                break
            }

            let neighborIDs = graph.neighborIDs(of: Int(current.nodeID))
            for neighborID in neighborIDs {
                let index = Int(neighborID)
                if neighborID == UInt32.max || index < 0 || index >= nodeCount {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)

                let candidateDistance = SIMDDistance.distance(
                    vector,
                    vectors.vector(at: index),
                    metric: metric
                )
                if results.count < efLimit || candidateDistance < results[results.count - 1].distance {
                    let candidate = Candidate(nodeID: neighborID, distance: candidateDistance)
                    insertSorted(candidate, into: &candidates)
                    insertSorted(candidate, into: &results)
                    if results.count > efLimit {
                        results.removeLast()
                    }
                }
            }
        }

        let selected = results.prefix(degree).filter { $0.nodeID != UInt32.max }
        return selected.map { ($0.nodeID, $0.distance) }
    }

    private static func worstNeighborIndex(in distances: [Float]) -> Int? {
        guard !distances.isEmpty else {
            return nil
        }

        var worstIndex = 0
        var worstDistance = distances[0]
        for index in 1..<distances.count {
            if distances[index] > worstDistance {
                worstDistance = distances[index]
                worstIndex = index
            }
        }
        return worstIndex
    }

    private static func insertSorted(_ candidate: Candidate, into list: inout [Candidate]) {
        var insertionIndex = list.endIndex
        for index in list.indices where candidate.distance < list[index].distance {
            insertionIndex = index
            break
        }
        list.insert(candidate, at: insertionIndex)
    }
}
