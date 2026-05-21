import Dispatch
import Foundation

public struct ProductQuantizer: Sendable, Codable {
    public let numSubspaces: Int
    public let centroidsPerSubspace: Int
    public let subspaceDimension: Int
    public let codebooks: [[[Float]]]

    public static func train(
        vectors: [[Float]],
        numSubspaces: Int = 8,
        centroidsPerSubspace: Int = 256,
        maxIterations: Int = 20
    ) throws -> ProductQuantizer {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot train ProductQuantizer with empty vectors")
        }
        guard numSubspaces > 0 else {
            throw ANNSError.constructionFailed("numSubspaces must be greater than zero")
        }
        guard centroidsPerSubspace == 256 else {
            throw ANNSError.constructionFailed("ProductQuantizer requires centroidsPerSubspace == 256")
        }

        let dimension = vectors[0].count
        guard dimension > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        guard vectors.count >= centroidsPerSubspace else {
            throw ANNSError.constructionFailed(
                "Need at least \(centroidsPerSubspace) vectors to train ProductQuantizer"
            )
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }
        guard dimension.isMultiple(of: numSubspaces) else {
            throw ANNSError.constructionFailed(
                "Vector dimension \(dimension) must be divisible by numSubspaces \(numSubspaces)"
            )
        }

        let subspaceDimension = dimension / numSubspaces
        let sharedState = ParallelTrainingState(slotCount: numSubspaces)

        DispatchQueue.concurrentPerform(iterations: numSubspaces) { subspace in
            guard sharedState.shouldContinue else {
                return
            }

            do {
                let result = try KMeans.clusterSubspace(
                    vectors: vectors,
                    offset: subspace * subspaceDimension,
                    dimension: subspaceDimension,
                    k: centroidsPerSubspace,
                    maxIterations: maxIterations,
                    metric: .l2
                )
                sharedState.store(result.centroids, at: subspace)
            } catch {
                sharedState.store(error: error)
            }
        }

        if let firstError = sharedState.firstError {
            throw firstError
        }

        let codebooks: [[[Float]]] = try sharedState.codebookSlots.enumerated().map { subspace, slot in
            guard let slot else {
                throw ANNSError.constructionFailed("Missing PQ codebook for subspace \(subspace)")
            }
            return slot
        }

        return ProductQuantizer(
            numSubspaces: numSubspaces,
            centroidsPerSubspace: centroidsPerSubspace,
            subspaceDimension: subspaceDimension,
            codebooks: codebooks
        )
    }

    public func encode(vector: [Float]) throws -> [UInt8] {
        let expectedDimension = numSubspaces * subspaceDimension
        guard vector.count == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: vector.count)
        }

        return vector.withUnsafeBufferPointer { vectorBuffer in
            guard let vectorBase = vectorBuffer.baseAddress else {
                return [UInt8](repeating: 0, count: numSubspaces)
            }

            var encoded = [UInt8](repeating: 0, count: numSubspaces)
            for subspace in 0..<numSubspaces {
                let centroids = codebooks[subspace]
                let queryBase = vectorBase + (subspace * subspaceDimension)

                var bestIndex = 0
                var bestDistance = Float.greatestFiniteMagnitude
                for centroidIndex in 0..<centroidsPerSubspace {
                    let distance = centroids[centroidIndex].withUnsafeBufferPointer { centroidBuffer in
                        SIMDDistance.l2(queryBase, centroidBuffer.baseAddress!, dim: subspaceDimension)
                    }
                    if distance < bestDistance {
                        bestDistance = distance
                        bestIndex = centroidIndex
                    }
                }
                encoded[subspace] = UInt8(bestIndex)
            }
            return encoded
        }
    }

    public func encode(vector: [Float], subtracting centroid: [Float]) throws -> [UInt8] {
        let expectedDimension = numSubspaces * subspaceDimension
        guard vector.count == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: vector.count)
        }
        guard centroid.count == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: centroid.count)
        }

        return vector.withUnsafeBufferPointer { vectorBuffer in
            centroid.withUnsafeBufferPointer { centroidBuffer in
                guard
                    let vectorBase = vectorBuffer.baseAddress,
                    let centroidBase = centroidBuffer.baseAddress
                else {
                    return [UInt8](repeating: 0, count: numSubspaces)
                }

                var encoded = [UInt8](repeating: 0, count: numSubspaces)
                for subspace in 0..<numSubspaces {
                    let centroids = codebooks[subspace]
                    let subspaceOffset = subspace * subspaceDimension
                    let queryBase = vectorBase + subspaceOffset
                    let subtractBase = centroidBase + subspaceOffset

                    var bestIndex = 0
                    var bestDistance = Float.greatestFiniteMagnitude
                    for centroidIndex in 0..<centroidsPerSubspace {
                        let distance = centroids[centroidIndex].withUnsafeBufferPointer { codebookBuffer in
                            var sum: Float = 0
                            let codebookBase = codebookBuffer.baseAddress!
                            for dimensionIndex in 0..<subspaceDimension {
                                let delta = queryBase[dimensionIndex] - subtractBase[dimensionIndex] - codebookBase[dimensionIndex]
                                sum += delta * delta
                            }
                            return sum
                        }
                        if distance < bestDistance {
                            bestDistance = distance
                            bestIndex = centroidIndex
                        }
                    }
                    encoded[subspace] = UInt8(bestIndex)
                }
                return encoded
            }
        }
    }

    public func reconstruct(codes: [UInt8]) throws -> [Float] {
        guard codes.count == numSubspaces else {
            throw ANNSError.dimensionMismatch(expected: numSubspaces, got: codes.count)
        }

        var vector: [Float] = []
        vector.reserveCapacity(numSubspaces * subspaceDimension)

        for subspace in 0..<numSubspaces {
            let centroidIndex = Int(codes[subspace])
            guard centroidIndex < centroidsPerSubspace else {
                throw ANNSError.constructionFailed("Code \(centroidIndex) out of range")
            }
            vector.append(contentsOf: codebooks[subspace][centroidIndex])
        }
        return vector
    }

    // FIXME: Major — approximateDistance sums sub-distances across PQ sub-spaces.
    // This decomposition is only mathematically valid for L2/squared-L2.
    // For cosine or innerProduct metrics the result is silently incorrect.
    public func approximateDistance(query: [Float], codes: [UInt8], metric: Metric) -> Float {
        guard let table = distanceTableFlattened(query: query, metric: metric), codes.count == numSubspaces else {
            return Float.greatestFiniteMagnitude
        }

        var distance: Float = 0
        for subspace in 0..<numSubspaces {
            let code = Int(codes[subspace])
            if code >= centroidsPerSubspace {
                return Float.greatestFiniteMagnitude
            }
            distance += table[(subspace * centroidsPerSubspace) + code]
        }
        return distance
    }

    package func distanceTable(query: [Float], metric: Metric) -> [[Float]]? {
        guard let flattened = distanceTableFlattened(query: query, metric: metric) else {
            return nil
        }

        var table = [[Float]]()
        table.reserveCapacity(numSubspaces)
        for subspace in 0..<numSubspaces {
            let start = subspace * centroidsPerSubspace
            let end = start + centroidsPerSubspace
            table.append(Array(flattened[start..<end]))
        }
        return table
    }

    package func distanceTableFlattened(query: [Float], metric: Metric) -> [Float]? {
        let expectedDimension = numSubspaces * subspaceDimension
        guard query.count == expectedDimension else {
            return nil
        }

        return query.withUnsafeBufferPointer { queryBuffer in
            guard let queryBase = queryBuffer.baseAddress else {
                return [Float]()
            }

            var table = [Float](repeating: 0, count: numSubspaces * centroidsPerSubspace)
            for subspace in 0..<numSubspaces {
                let centroids = codebooks[subspace]
                let subQueryBase = queryBase + (subspace * subspaceDimension)

                for centroidIndex in 0..<centroidsPerSubspace {
                    table[(subspace * centroidsPerSubspace) + centroidIndex] =
                        centroids[centroidIndex].withUnsafeBufferPointer { centroidBuffer in
                            SIMDDistance.distance(
                                subQueryBase,
                                centroidBuffer.baseAddress!,
                                dim: subspaceDimension,
                                metric: metric
                            )
                        }
                }
            }
            return table
        }
    }
}

// Synchronized via NSLock; all mutable access is protected.
private final class ParallelTrainingState: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var codebookSlots: [[[Float]]?]
    private(set) var firstError: Error?

    init(slotCount: Int) {
        codebookSlots = Array(repeating: nil, count: slotCount)
    }

    var shouldContinue: Bool {
        lock.lock()
        defer { lock.unlock() }
        return firstError == nil
    }

    func store(_ codebook: [[Float]], at index: Int) {
        lock.lock()
        codebookSlots[index] = codebook
        lock.unlock()
    }

    func store(error: Error) {
        lock.lock()
        if firstError == nil {
            firstError = error
        }
        lock.unlock()
    }
}
