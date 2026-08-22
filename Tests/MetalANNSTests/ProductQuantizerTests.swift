import Foundation
import Testing

@testable import MetalANNSCore

@Suite("ProductQuantizer Tests")
struct ProductQuantizerTests {
    @Test("Train PQ codebook on 1000 vectors without errors")
    func trainPQCodebook() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 11)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        #expect(pq.numSubspaces == 8)
        #expect(pq.centroidsPerSubspace == 256)
        #expect(pq.subspaceDimension == 16)
        #expect(pq.codebooks.count == 8)
        #expect(pq.codebooks.allSatisfy { $0.count == 256 })
    }

    @Test("Encode vectors into M-byte codes")
    func encodeVectors() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 33)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        for vector in vectors.prefix(100) {
            let codes = try pq.encode(vector: vector)
            #expect(codes.count == 8)
            #expect(codes.allSatisfy { $0 <= UInt8.max })
        }
    }

    @Test("Encode then reconstruct with low relative error")
    func reconstructionAccuracy() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 71)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        var maxRelativeError: Float = 0
        for vector in vectors.prefix(100) {
            let codes = try pq.encode(vector: vector)
            let reconstructed = try pq.reconstruct(codes: codes)
            let relativeError = l2RelativeError(original: vector, reconstructed: reconstructed)
            maxRelativeError = max(maxRelativeError, relativeError)
        }

        #expect(maxRelativeError < 0.02, "Max relative reconstruction error \(maxRelativeError)")
    }

    @Test("Approximate distances are highly correlated with exact distances")
    func distanceApproximationAccuracy() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 101)
        let queries = makeStructuredVectors(count: 32, dimension: 128, seed: 202)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        let codes = try vectors.prefix(256).map { try pq.encode(vector: $0) }
        var exactDistances: [Float] = []
        var approximateDistances: [Float] = []
        exactDistances.reserveCapacity(queries.count * codes.count)
        approximateDistances.reserveCapacity(queries.count * codes.count)

        for query in queries {
            for (index, code) in codes.enumerated() {
                approximateDistances.append(pq.approximateDistance(query: query, codes: code, metric: .l2))
                exactDistances.append(SIMDDistance.distance(query, vectors[index], metric: .l2))
            }
        }

        let correlation = pearsonCorrelation(exactDistances, approximateDistances)
        #expect(correlation > 0.95, "Observed correlation \(correlation)")
    }

    @Test("Encoding with centroid subtraction matches encoding residual vector directly")
    func encodeSubtractingCentroidMatchesResidual() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 303)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        let vector = vectors[17]
        let centroid = vectors[41]
        let residual = zip(vector, centroid).map { $0 - $1 }

        let direct = try pq.encode(vector: residual)
        let subtracting = try pq.encode(vector: vector, subtracting: centroid)

        #expect(direct == subtracting)
    }
}

private func l2RelativeError(original: [Float], reconstructed: [Float]) -> Float {
    let originalNorm = original.reduce(Float(0)) { $0 + ($1 * $1) }.squareRoot()
    let diffNorm = zip(original, reconstructed).reduce(Float(0)) { partial, pair in
        let delta = pair.0 - pair.1
        return partial + (delta * delta)
    }.squareRoot()
    return diffNorm / max(originalNorm, 1e-8)
}

private func pearsonCorrelation(_ x: [Float], _ y: [Float]) -> Float {
    precondition(x.count == y.count)
    guard x.count > 1 else {
        return 1
    }

    let count = Float(x.count)
    let meanX = x.reduce(0, +) / count
    let meanY = y.reduce(0, +) / count

    var numerator: Float = 0
    var varianceX: Float = 0
    var varianceY: Float = 0
    for (xi, yi) in zip(x, y) {
        let dx = xi - meanX
        let dy = yi - meanY
        numerator += dx * dy
        varianceX += dx * dx
        varianceY += dy * dy
    }

    let denom = (varianceX * varianceY).squareRoot()
    guard denom > 1e-12 else {
        return 0
    }
    return numerator / denom
}

private func makeStructuredVectors(count: Int, dimension: Int, seed: UInt64) -> [[Float]] {
    precondition(dimension == 128)
    let numSubspaces = 8
    let subspaceDimension = 16
    let prototypeCount = 64

    var prototypes = [[[Float]]](repeating: [[Float]](), count: numSubspaces)
    for subspace in 0..<numSubspaces {
        prototypes[subspace] = (0..<prototypeCount).map { proto in
            (0..<subspaceDimension).map { d in
                let base = Float((proto * 31 + d * 17 + subspace * 13) % 97) / 97.0
                return base + Float(subspace) * 0.07
            }
        }
    }

    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<count).map { row in
        var vector: [Float] = []
        vector.reserveCapacity(dimension)
        for subspace in 0..<numSubspaces {
            let protoIndex = (row * 7 + subspace * 11 + Int(rng.next() % UInt64(prototypeCount))) % prototypeCount
            for value in prototypes[subspace][protoIndex] {
                let noise = Float.random(in: -0.004...0.004, using: &rng)
                vector.append(value + noise)
            }
        }
        return vector
    }
}
