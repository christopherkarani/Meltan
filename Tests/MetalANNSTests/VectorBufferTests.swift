import Testing

@testable import MetalANNSCore

@Suite("VectorBuffer Tests")
struct VectorBufferTests {
    @Test("Insert and read back single vector")
    func insertSingle() throws {
        let buffer = try VectorBuffer(capacity: 10, dim: 3)
        let vector: [Float] = [1.0, 2.0, 3.0]

        try buffer.insert(vector: vector, at: 0)
        let readBack = buffer.vector(at: 0)

        #expect(readBack == vector)
    }

    @Test("Batch insert 100 vectors and verify roundtrip")
    func batchInsert() throws {
        let dim = 128
        let vectorCount = 100
        let buffer = try VectorBuffer(capacity: vectorCount, dim: dim)
        var vectors = [[Float]]()

        for _ in 0..<vectorCount {
            vectors.append(seededVector(dim: dim))
        }

        try buffer.batchInsert(vectors: vectors, startingAt: 0)

        for index in 0..<vectorCount {
            let readBack = buffer.vector(at: index)
            for dimension in 0..<dim {
                #expect(abs(readBack[dimension] - vectors[index][dimension]) < 1e-7)
            }
        }
    }

    @Test("Count tracks insertions")
    func countTracking() throws {
        let buffer = try VectorBuffer(capacity: 10, dim: 4)

        #expect(buffer.count == 0)
        try buffer.insert(vector: [1, 2, 3, 4], at: 0)
        buffer.setCount(1)
        #expect(buffer.count == 1)
    }
}
