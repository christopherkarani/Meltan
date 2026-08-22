import Accelerate
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Distance Computation Tests")
struct DistanceTests {
    let backend = AccelerateBackend()

    private func withVectorBuffer<T>(
        _ values: [Float],
        _ body: (UnsafeBufferPointer<Float>) async throws -> T
    ) async throws -> T {
        let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
        buffer.initialize(from: values)
        defer {
            buffer.deinitialize()
            buffer.deallocate()
        }
        return try await body(UnsafeBufferPointer(buffer))
    }

    @Test("Cosine distance of identical vectors is 0")
    func cosineIdentical() async throws {
        let v = [Float](repeating: 1.0, count: 128)
        let distances = try await withVectorBuffer(v) { ptr in
            try await backend.computeDistances(
                query: v,
                vectors: ptr,
                vectorCount: 1,
                dim: 128,
                metric: .cosine
            )
        }
        #expect(distances.count == 1)
        #expect(abs(distances[0]) < 1e-5)
    }

    @Test("Cosine distance of orthogonal vectors is 1")
    func cosineOrthogonal() async throws {
        var v1 = [Float](repeating: 0, count: 4)
        v1[0] = 1.0
        var v2 = [Float](repeating: 0, count: 4)
        v2[1] = 1.0
        let distances = try await withVectorBuffer(v2) { ptr in
            try await backend.computeDistances(
                query: v1,
                vectors: ptr,
                vectorCount: 1,
                dim: 4,
                metric: .cosine
            )
        }
        #expect(abs(distances[0] - 1.0) < 1e-5)
    }

    @Test("L2 distance of identical vectors is 0")
    func l2Identical() async throws {
        let v = [Float](repeating: 1.0, count: 128)
        let distances = try await withVectorBuffer(v) { ptr in
            try await backend.computeDistances(
                query: v,
                vectors: ptr,
                vectorCount: 1,
                dim: 128,
                metric: .l2
            )
        }
        #expect(abs(distances[0]) < 1e-5)
    }

    @Test("L2 distance is squared Euclidean")
    func l2Squared() async throws {
        let q: [Float] = [1, 0, 0]
        let v: [Float] = [0, 1, 0]
        let distances = try await withVectorBuffer(v) { ptr in
            try await backend.computeDistances(
                query: q,
                vectors: ptr,
                vectorCount: 1,
                dim: 3,
                metric: .l2
            )
        }
        #expect(abs(distances[0] - 2.0) < 1e-5)
    }

    @Test("Inner product of unit vectors")
    func innerProduct() async throws {
        let q: [Float] = [1, 0, 0]
        let v: [Float] = [0.5, 0.5, 0]
        let distances = try await withVectorBuffer(v) { ptr in
            try await backend.computeDistances(
                query: q,
                vectors: ptr,
                vectorCount: 1,
                dim: 3,
                metric: .innerProduct
            )
        }
        #expect(abs(distances[0] - (-0.5)) < 1e-5)
    }

    @Test("Batch distances: 1000 random 128-dim vectors")
    func batchDistances() async throws {
        let dim = 128
        let n = 1000
        var vectors = [Float](repeating: 0, count: n * dim)
        fillWithSeededRandom(&vectors)
        var query = [Float](repeating: 0, count: dim)
        fillWithSeededRandom(&query, seed: 43)

        let distances = try await withVectorBuffer(vectors) { ptr in
            try await backend.computeDistances(
                query: query,
                vectors: ptr,
                vectorCount: n,
                dim: dim,
                metric: .cosine
            )
        }
        #expect(distances.count == n)
        for d in distances {
            #expect(d >= -1e-5 && d <= 2.0 + 1e-5)
        }
    }

    @Test("Edge case: dim=1")
    func dim1() async throws {
        let q: [Float] = [3.0]
        let v: [Float] = [4.0]
        let distances = try await withVectorBuffer(v) { ptr in
            try await backend.computeDistances(
                query: q,
                vectors: ptr,
                vectorCount: 1,
                dim: 1,
                metric: .l2
            )
        }
        #expect(abs(distances[0] - 1.0) < 1e-5)
    }

    @Test("Edge case: dim=1536 (large embedding)")
    func dimLarge() async throws {
        let dim = 1536
        let v = [Float](repeating: 1.0 / sqrt(Float(dim)), count: dim)
        let distances = try await withVectorBuffer(v) { ptr in
            try await backend.computeDistances(
                query: v,
                vectors: ptr,
                vectorCount: 1,
                dim: dim,
                metric: .cosine
            )
        }
        #expect(abs(distances[0]) < 1e-4)
    }
}
