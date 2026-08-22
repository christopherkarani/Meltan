import Foundation
import Testing

@testable import MetalANNSCore

@Suite("PQVectorBuffer Tests")
struct PQVectorBufferTests {
    @Test("Initialize and insert encoded vectors")
    func initAndInsert() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 41)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        let buffer = try PQVectorBuffer(capacity: 200, dim: 128, pq: pq)
        for (index, vector) in vectors.prefix(100).enumerated() {
            try buffer.insert(vector: vector, at: index)
        }

        #expect(buffer.count == 100)
        #expect(buffer.originalDimension == 128)
        #expect(buffer.reconstruct(at: 0).count == 128)
    }

    @Test("Approximate distance is deterministic and finite")
    func approximateDistance() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 73)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        let buffer = try PQVectorBuffer(capacity: 128, dim: 128, pq: pq)
        for (index, vector) in vectors.prefix(128).enumerated() {
            try buffer.insert(vector: vector, at: index)
        }

        let query = vectors[700]
        let d1 = buffer.approximateDistance(query: query, to: 42, metric: .l2)
        let d2 = buffer.approximateDistance(query: query, to: 42, metric: .l2)
        let d3 = buffer.approximateDistance(query: query, to: 63, metric: .l2)

        #expect(d1.isFinite)
        #expect(d2.isFinite)
        #expect(d3.isFinite)
        #expect(abs(d1 - d2) < 1e-7)
    }

    @Test("PQ storage yields 30x+ memory reduction")
    func memoryReduction() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 99)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        let vectorCount = 500
        let buffer = try PQVectorBuffer(capacity: vectorCount, dim: 128, pq: pq)
        for (index, vector) in vectors.prefix(vectorCount).enumerated() {
            try buffer.insert(vector: vector, at: index)
        }

        let uncompressedBytes = vectorCount * 128 * MemoryLayout<Float>.stride
        let compressedBytes = vectorCount * pq.numSubspaces * MemoryLayout<UInt8>.stride
        let reduction = Float(uncompressedBytes) / Float(compressedBytes)

        #expect(reduction >= 30)
        #expect(reduction <= 70)
    }

    @Test("Gathered codes preserve requested row order")
    func gatherCodesPreservesOrder() throws {
        let vectors = makeStructuredVectors(count: 1_000, dimension: 128, seed: 123)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 8,
            centroidsPerSubspace: 256,
            maxIterations: 8
        )

        let buffer = try PQVectorBuffer(capacity: 8, dim: 128, pq: pq)
        for index in 0..<4 {
            try buffer.insert(vector: vectors[index], at: index)
        }

        let ids: [UInt32] = [2, 0, 3]
        let gathered = buffer.gatherCodes(for: ids)

        #expect(gathered.count == ids.count * pq.numSubspaces)
        for (row, id) in ids.enumerated() {
            let start = row * pq.numSubspaces
            let end = start + pq.numSubspaces
            #expect(Array(gathered[start..<end]) == buffer.code(at: Int(id)))
        }
    }
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
