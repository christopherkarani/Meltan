import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "NNDescentCPU")

package enum NNDescentCPU {
    package static func build(
        vectors: [[Float]],
        degree: Int,
        metric: Metric,
        maxIterations: Int = 20,
        convergenceThreshold: Float = 0.001
    ) async throws -> (graph: [[(UInt32, Float)]], entryPoint: UInt32) {
        let nodeCount = vectors.count
        guard nodeCount > 1 else {
            throw ANNSError.constructionFailed("NNDescentCPU requires at least 2 vectors")
        }
        guard degree > 0, degree < nodeCount else {
            throw ANNSError.constructionFailed("Degree must be in 1..<(nodeCount)")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.constructionFailed("Vector dimension must be greater than 0")
        }
        for vector in vectors {
            if vector.count != dim {
                throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
            }
        }

        @inline(__always)
        func distance(_ lhs: Int, _ rhs: Int) -> Float {
            SIMDDistance.distance(vectors[lhs], vectors[rhs], metric: metric)
        }

        struct LCG {
            var state: UInt64

            mutating func next(upperBound: Int) -> Int {
                state = state &* 6_364_136_223_846_793_005 &+ 1
                return Int(state % UInt64(upperBound))
            }
        }

        typealias Neighbor = (UInt32, Float)
        var graph = [[Neighbor]](repeating: [], count: nodeCount)
        var marks = [Int](repeating: -1, count: nodeCount)
        var candidateScratch: [Int] = []

        for node in 0..<nodeCount {
            var generator = LCG(state: UInt64(node + 1) &* 0x9E37_79B9_7F4A_7C15)
            var neighbors = Set<Int>()
            while neighbors.count < degree {
                let candidate = generator.next(upperBound: nodeCount)
                if candidate != node {
                    neighbors.insert(candidate)
                }
            }

            var list = [Neighbor]()
            list.reserveCapacity(degree)
            for candidate in neighbors {
                list.append((UInt32(candidate), distance(node, candidate)))
            }
            list.sort { $0.1 < $1.1 }
            graph[node] = list
        }

        func tryInsert(_ candidate: Int, into node: Int, dist: Float, graph: inout [[Neighbor]]) -> Bool {
            if candidate == node {
                return false
            }

            var neighbors = graph[node]
            if neighbors.contains(where: { Int($0.0) == candidate }) {
                return false
            }

            if neighbors.count < degree {
                neighbors.append((UInt32(candidate), dist))
                graph[node] = neighbors
                return true
            }

            var worstIndex = 0
            var worstDistance = neighbors[0].1
            for idx in 1..<neighbors.count where neighbors[idx].1 > worstDistance {
                worstDistance = neighbors[idx].1
                worstIndex = idx
            }

            if dist >= worstDistance {
                return false
            }

            neighbors[worstIndex] = (UInt32(candidate), dist)
            graph[node] = neighbors
            return true
        }

        for iteration in 0..<maxIterations {
            var reverse = [[Int]](repeating: [], count: nodeCount)
            for source in 0..<nodeCount {
                for (targetID, _) in graph[source] {
                    let target = Int(targetID)
                    if target < nodeCount {
                        reverse[target].append(source)
                    }
                }
            }

            var updateCount = 0

            for node in 0..<nodeCount {
                let markToken = ((iteration + 1) * nodeCount) + node
                candidateScratch.removeAll(keepingCapacity: true)
                for (neighborID, _) in graph[node] {
                    let neighbor = Int(neighborID)
                    guard neighbor >= 0, neighbor < nodeCount, marks[neighbor] != markToken else {
                        continue
                    }
                    marks[neighbor] = markToken
                    candidateScratch.append(neighbor)
                }
                for neighbor in reverse[node] where neighbor >= 0 && neighbor < nodeCount {
                    guard marks[neighbor] != markToken else {
                        continue
                    }
                    marks[neighbor] = markToken
                    candidateScratch.append(neighbor)
                }

                if candidateScratch.count < 2 {
                    continue
                }

                for i in 0..<(candidateScratch.count - 1) {
                    let a = candidateScratch[i]
                    for j in (i + 1)..<candidateScratch.count {
                        let b = candidateScratch[j]
                        if a == b {
                            continue
                        }

                        let dist = distance(a, b)
                        if tryInsert(b, into: a, dist: dist, graph: &graph) {
                            updateCount += 1
                        }
                        if tryInsert(a, into: b, dist: dist, graph: &graph) {
                            updateCount += 1
                        }
                    }
                }
            }

            logger.debug("NNDescentCPU iteration \(iteration): \(updateCount) updates")

            if Float(updateCount) < convergenceThreshold * Float(degree * nodeCount) {
                logger.debug("NNDescentCPU converged at iteration \(iteration + 1)")
                break
            }
        }

        var entryPoint: UInt32 = 0
        var bestMean = Float.greatestFiniteMagnitude

        for node in 0..<nodeCount {
            graph[node].sort { $0.1 < $1.1 }
        }

        for node in 0..<nodeCount {
            let mean = graph[node].reduce(Float(0)) { $0 + $1.1 } / Float(degree)
            if mean < bestMean {
                bestMean = mean
                entryPoint = UInt32(node)
            }
        }

        return (graph, entryPoint)
    }
}
