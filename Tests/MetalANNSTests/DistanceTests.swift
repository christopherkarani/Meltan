import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Distance Computation Tests")
struct DistanceTests {
    @Test("Cosine distance of identical vectors is 0")
    func cosineIdentical() {
        let v = [Float](repeating: 1.0, count: 128)
        let distance = SIMDDistance.distance(v, v, metric: .cosine)
        #expect(abs(distance) < 1e-5)
    }

    @Test("Cosine distance of orthogonal vectors is 1")
    func cosineOrthogonal() {
        var v1 = [Float](repeating: 0, count: 4)
        v1[0] = 1.0
        var v2 = [Float](repeating: 0, count: 4)
        v2[1] = 1.0
        let distance = SIMDDistance.distance(v1, v2, metric: .cosine)
        #expect(abs(distance - 1.0) < 1e-5)
    }

    @Test("L2 distance of identical vectors is 0")
    func l2Identical() {
        let v = [Float](repeating: 1.0, count: 128)
        let distance = SIMDDistance.distance(v, v, metric: .l2)
        #expect(abs(distance) < 1e-5)
    }

    @Test("L2 distance is squared Euclidean")
    func l2Squared() {
        let q: [Float] = [1, 0, 0]
        let v: [Float] = [0, 1, 0]
        let distance = SIMDDistance.distance(q, v, metric: .l2)
        #expect(abs(distance - 2.0) < 1e-5)
    }

    @Test("Inner product of unit vectors")
    func innerProduct() {
        let q: [Float] = [1, 0, 0]
        let v: [Float] = [0.5, 0.5, 0]
        let distance = SIMDDistance.distance(q, v, metric: .innerProduct)
        #expect(abs(distance - (-0.5)) < 1e-5)
    }

    @Test("Batch distances: 1000 random 128-dim vectors")
    func batchDistances() {
        let dim = 128
        let n = 1000
        var flatVectors = [Float](repeating: 0, count: n * dim)
        fillWithSeededRandom(&flatVectors)
        var query = [Float](repeating: 0, count: dim)
        fillWithSeededRandom(&query, seed: 43)

        let distances = (0..<n).map { index -> Float in
            let start = index * dim
            return SIMDDistance.distance(
                Array(flatVectors[start..<start + dim]),
                query,
                metric: .cosine
            )
        }
        #expect(distances.count == n)
        for d in distances {
            #expect(d >= -1e-5 && d <= 2.0 + 1e-5)
        }
    }

    @Test("Edge case: dim=1")
    func dim1() {
        let q: [Float] = [3.0]
        let v: [Float] = [4.0]
        let distance = SIMDDistance.distance(q, v, metric: .l2)
        #expect(abs(distance - 1.0) < 1e-5)
    }

    @Test("Edge case: dim=1536 (large embedding)")
    func dimLarge() {
        let dim = 1536
        let v = [Float](repeating: 1.0 / sqrt(Float(dim)), count: dim)
        let distance = SIMDDistance.distance(v, v, metric: .cosine)
        #expect(abs(distance) < 1e-4)
    }
}
