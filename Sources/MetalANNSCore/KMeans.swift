import Dispatch
import Foundation

public enum KMeans {
    public struct Result {
        public let centroids: [[Float]]
        public let assignments: [Int]

        public init(centroids: [[Float]], assignments: [Int]) {
            self.centroids = centroids
            self.assignments = assignments
        }
    }

    public static func cluster(
        vectors: [[Float]],
        k: Int,
        maxIterations: Int = 20,
        metric: Metric = .l2,
        seed: UInt64 = 42
    ) throws -> Result {
        try cluster(
            vectors: vectors,
            offset: 0,
            dimension: nil,
            k: k,
            maxIterations: maxIterations,
            metric: metric,
            seed: seed
        )
    }

    static func clusterSubspace(
        vectors: [[Float]],
        offset: Int,
        dimension: Int,
        k: Int,
        maxIterations: Int = 20,
        metric: Metric = .l2,
        seed: UInt64 = 42
    ) throws -> Result {
        try cluster(
            vectors: vectors,
            offset: offset,
            dimension: dimension,
            k: k,
            maxIterations: maxIterations,
            metric: metric,
            seed: seed
        )
    }

    private static func cluster(
        vectors: [[Float]],
        offset: Int,
        dimension: Int?,
        k: Int,
        maxIterations: Int,
        metric: Metric,
        seed: UInt64
    ) throws -> Result {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot cluster empty vectors")
        }
        guard k > 0, k <= vectors.count else {
            throw ANNSError.constructionFailed("k must be between 1 and vector count")
        }

        let dim = dimension ?? vectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        for vector in vectors {
            let vectorDimension = vector.count
            if vectorDimension < offset + dim {
                throw ANNSError.dimensionMismatch(expected: offset + dim, got: vectorDimension)
            }
        }

        var centroids = initializeCentroids(
            vectors: vectors,
            offset: offset,
            dim: dim,
            k: k,
            metric: metric,
            seed: seed
        )
        var assignments = [Int](repeating: -1, count: vectors.count)
        var centroidSums = [Float](repeating: 0, count: k * dim)
        var counts = [Int](repeating: 0, count: k)
        let workerCount = min(max(1, ProcessInfo.processInfo.activeProcessorCount), vectors.count)
        let useParallelLloyd = workerCount > 1 && vectors.count >= 512 && k >= 8

        for _ in 0..<maxIterations {
            centroidSums.withUnsafeMutableBufferPointer { sumsBuffer in
                sumsBuffer.initialize(repeating: 0)
            }
            counts.withUnsafeMutableBufferPointer { countsBuffer in
                countsBuffer.initialize(repeating: 0)
            }

            let changed: Bool
            if useParallelLloyd {
                let chunkSize = (vectors.count + workerCount - 1) / workerCount
                var nextAssignments = [Int](repeating: -1, count: vectors.count)
                let previousAssignments = assignments
                let flatCentroids = centroids.flatMap { $0 }
                let workerStates = (0..<workerCount).map { _ in
                    LloydWorkerState(clusterCount: k, dimension: dim)
                }

                DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
                    let start = worker * chunkSize
                    let end = min(vectors.count, start + chunkSize)
                    guard start < end else {
                        return
                    }

                    let state = workerStates[worker]
                    state.reset(assignmentCount: end - start)

                    for i in start..<end {
                        let vector = vectors[i]
                        let bestCluster = vector.withUnsafeBufferPointer { vectorBuffer in
                            let vectorBase = vectorBuffer.baseAddress! + offset
                            var bestCluster = 0
                            var bestDistance = Float.greatestFiniteMagnitude

                            for c in 0..<k {
                                let centroidOffset = c * dim
                                let distance = flatCentroids.withUnsafeBufferPointer { centroidBuffer in
                                    SIMDDistance.distance(
                                        vectorBase,
                                        centroidBuffer.baseAddress!.advanced(by: centroidOffset),
                                        dim: dim,
                                        metric: metric
                                    )
                                }
                                if distance < bestDistance {
                                    bestDistance = distance
                                    bestCluster = c
                                }
                            }

                            return bestCluster
                        }

                        state.assignments[i - start] = bestCluster
                        if previousAssignments[i] != bestCluster {
                            state.changed = true
                        }

                        vector.withUnsafeBufferPointer { vectorBuffer in
                            let vectorBase = vectorBuffer.baseAddress! + offset
                            state.counts[bestCluster] += 1
                            let sumOffset = bestCluster * dim
                            for d in 0..<dim {
                                state.sums[sumOffset + d] += vectorBase[d]
                            }
                        }
                    }
                }

                changed = workerStates.contains(where: \.changed)
                if !changed {
                    break
                }

                for worker in 0..<workerCount {
                    let start = worker * chunkSize
                    let end = min(vectors.count, start + chunkSize)
                    guard start < end else {
                        continue
                    }

                    let state = workerStates[worker]
                    for localIndex in 0..<(end - start) {
                        nextAssignments[start + localIndex] = state.assignments[localIndex]
                    }
                    for cluster in 0..<k {
                        counts[cluster] += state.counts[cluster]
                        let clusterOffset = cluster * dim
                        for d in 0..<dim {
                            centroidSums[clusterOffset + d] += state.sums[clusterOffset + d]
                        }
                    }
                }
                assignments = nextAssignments
            } else {
                var sequentialChanged = false

                for i in 0..<vectors.count {
                    let vector = vectors[i]
                    let bestCluster = centroids.withUnsafeBufferPointer { centroidRows in
                        vector.withUnsafeBufferPointer { vectorBuffer in
                            let vectorBase = vectorBuffer.baseAddress! + offset
                            var bestCluster = 0
                            var bestDistance = Float.greatestFiniteMagnitude

                            for c in 0..<k {
                                let distance = centroidRows[c].withUnsafeBufferPointer { centroidBuffer in
                                    SIMDDistance.distance(
                                        vectorBase,
                                        centroidBuffer.baseAddress!,
                                        dim: dim,
                                        metric: metric
                                    )
                                }
                                if distance < bestDistance {
                                    bestDistance = distance
                                    bestCluster = c
                                }
                            }

                            return bestCluster
                        }
                    }

                    if assignments[i] != bestCluster {
                        assignments[i] = bestCluster
                        sequentialChanged = true
                    }

                    counts[bestCluster] += 1
                    vectors[i].withUnsafeBufferPointer { vectorBuffer in
                        let vectorBase = vectorBuffer.baseAddress! + offset
                        let sumOffset = bestCluster * dim
                        for d in 0..<dim {
                            centroidSums[sumOffset + d] += vectorBase[d]
                        }
                    }
                }

                changed = sequentialChanged
                if !changed {
                    break
                }
            }

            for c in 0..<k {
                guard counts[c] > 0 else {
                    // FIXME: Major — empty cluster leaves stale centroid.
                    // Standard remedy is to reinitialize (e.g., pick farthest point).
                    // Stale centroids reduce effective K below requested value.
                    continue
                }
                let scale = 1.0 / Float(counts[c])
                let sumOffset = c * dim
                for d in 0..<dim {
                    centroids[c][d] = centroidSums[sumOffset + d] * scale
                }
            }
        }

        return Result(centroids: centroids, assignments: assignments)
    }

    private static func initializeCentroids(
        vectors: [[Float]],
        offset: Int,
        dim: Int,
        k: Int,
        metric: Metric,
        seed: UInt64
    ) -> [[Float]] {
        var rng = RandomNumberGenerator64(seed: seed)
        var centroids: [[Float]] = []
        centroids.reserveCapacity(k)
        var minDistances = [Float](repeating: Float.greatestFiniteMagnitude, count: vectors.count)

        let firstIndex = Int.random(in: 0..<vectors.count, using: &rng)
        centroids.append(Array(vectors[firstIndex][offset..<(offset + dim)]))
        updateMinDistances(
            vectors: vectors,
            offset: offset,
            dim: dim,
            metric: metric,
            newestCentroid: centroids[0],
            minDistances: &minDistances
        )

        for _ in 1..<k {
            var totalDistance: Float = 0
            for distance in minDistances {
                totalDistance += distance
            }

            if totalDistance <= 0 {
                let randomIndex = Int.random(in: 0..<vectors.count, using: &rng)
                let centroid = Array(vectors[randomIndex][offset..<(offset + dim)])
                centroids.append(centroid)
                updateMinDistances(
                    vectors: vectors,
                    offset: offset,
                    dim: dim,
                    metric: metric,
                    newestCentroid: centroid,
                    minDistances: &minDistances
                )
                continue
            }

            var threshold = Float.random(in: 0..<totalDistance, using: &rng)
            var chosenIndex = vectors.count - 1

            for i in 0..<vectors.count {
                threshold -= minDistances[i]
                if threshold <= 0 {
                    chosenIndex = i
                    break
                }
            }

            let centroid = Array(vectors[chosenIndex][offset..<(offset + dim)])
            centroids.append(centroid)
            updateMinDistances(
                vectors: vectors,
                offset: offset,
                dim: dim,
                metric: metric,
                newestCentroid: centroid,
                minDistances: &minDistances
            )
        }

        return centroids
    }

    private static func updateMinDistances(
        vectors: [[Float]],
        offset: Int,
        dim: Int,
        metric: Metric,
        newestCentroid: [Float],
        minDistances: inout [Float]
    ) {
        newestCentroid.withUnsafeBufferPointer { centroidBuffer in
            let centroidBase = centroidBuffer.baseAddress!
            for index in 0..<vectors.count {
                vectors[index].withUnsafeBufferPointer { vectorBuffer in
                    let vectorBase = vectorBuffer.baseAddress! + offset
                    let distance = SIMDDistance.distance(
                        vectorBase,
                        centroidBase,
                        dim: dim,
                        metric: metric
                    )
                    if distance < minDistances[index] {
                        minDistances[index] = distance
                    }
                }
            }
        }
    }
}

private struct RandomNumberGenerator64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 1 : seed
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

// Synchronized via DispatchQueue.concurrentPerform barrier;
// each worker gets its own state instance.
private final class LloydWorkerState: @unchecked Sendable {
    var assignments: [Int] = []
    var sums: [Float]
    var counts: [Int]
    var changed = false

    init(clusterCount: Int, dimension: Int) {
        sums = [Float](repeating: 0, count: clusterCount * dimension)
        counts = [Int](repeating: 0, count: clusterCount)
    }

    func reset(assignmentCount: Int) {
        assignments = [Int](repeating: -1, count: assignmentCount)
        sums.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
        counts.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
        changed = false
    }
}
