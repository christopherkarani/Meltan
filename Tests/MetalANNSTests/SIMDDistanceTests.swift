import Foundation
import Testing
import MetalANNSCore

@Suite("SIMD Distance Tests")
struct SIMDDistanceTests {
    @Test("cosineMatchesScalar")
    func cosineMatchesScalar() {
        let dim = 128
        let a = randomVector(dim: dim)
        let b = randomVector(dim: dim)

        let simd = SIMDDistance.cosine(a, b)
        let scalar = scalarCosine(a, b)
        #expect(abs(simd - scalar) < 1e-5)
    }

    @Test("l2MatchesScalar")
    func l2MatchesScalar() {
        let dim = 128
        let a = randomVector(dim: dim)
        let b = randomVector(dim: dim)

        let simd = SIMDDistance.l2(a, b)
        let scalar = scalarL2(a, b)
        #expect(abs(simd - scalar) < 1e-5)
    }

    @Test("innerProductMatchesScalar")
    func innerProductMatchesScalar() {
        let dim = 128
        let a = randomVector(dim: dim)
        let b = randomVector(dim: dim)

        let simd = SIMDDistance.innerProduct(a, b)
        let scalar = scalarInnerProduct(a, b)
        #expect(abs(simd - scalar) < 1e-5)
    }

    @Test("hammingMatchesScalar")
    func hammingMatchesScalar() {
        let a: [Float] = [1, 0, 1, 0, 1, 1, 0, 0]
        let b: [Float] = [0, 0, 1, 1, 1, 0, 0, 1]

        let simd = SIMDDistance.hamming(a, b)
        let scalar = scalarHamming(a, b)
        #expect(simd == scalar)
        #expect(SIMDDistance.distance(a, b, metric: .hamming) == scalar)
    }

    @Test("packedHammingMatchesScalar")
    func packedHammingMatchesScalar() {
        let a: [Float] = [1, 0, 1, 0, 1, 1, 0, 0]
        let b: [Float] = [0, 0, 1, 1, 1, 0, 0, 1]

        let packedA = packBinary(a)
        let packedB = packBinary(b)

        let packedDistance = SIMDDistance.hamming(packed: packedA, packed: packedB)
        let scalar = scalarHamming(a, b)
        #expect(packedDistance == scalar)
    }

    @Test("simdFasterThanScalar")
    func simdFasterThanScalar() {
        let dim = 256
        let iterations = 10_000
        let a = randomVector(dim: dim)
        let b = randomVector(dim: dim)
        let clock = ContinuousClock()

        var warmup: Float = 0
        for _ in 0..<1_000 {
            warmup += SIMDDistance.cosine(a, b)
            warmup += scalarCosine(a, b)
        }
        #expect(warmup.isFinite)

        var simdAccumulator: Float = 0
        let simdStart = clock.now
        for _ in 0..<iterations {
            simdAccumulator += SIMDDistance.cosine(a, b)
        }
        let simdTime = simdStart.duration(to: clock.now)
        #expect(simdAccumulator.isFinite)

        var scalarAccumulator: Float = 0
        let scalarStart = clock.now
        for _ in 0..<iterations {
            scalarAccumulator += scalarCosine(a, b)
        }
        let scalarTime = scalarStart.duration(to: clock.now)
        #expect(scalarAccumulator.isFinite)

        #expect(simdTime < scalarTime)
    }

    private func randomVector(dim: Int) -> [Float] {
        var rng = SeededGenerator(state: 42)
        return (0..<dim).map { _ in Float.random(in: -0.25...0.25, using: &rng) }
    }

    private func scalarCosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            let lhs = a[i]
            let rhs = b[i]
            dot += lhs * rhs
            normA += lhs * lhs
            normB += rhs * rhs
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
    }

    private func scalarL2(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sum
    }

    private func scalarInnerProduct(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
        }
        return -dot
    }

    private func scalarHamming(_ a: [Float], _ b: [Float]) -> Float {
        var count = 0
        for index in 0..<a.count where a[index] != b[index] {
            count += 1
        }
        return Float(count)
    }

    private func packBinary(_ vector: [Float]) -> [UInt8] {
        precondition(vector.count % 8 == 0)
        let bytes = vector.count / 8
        var packed = [UInt8](repeating: 0, count: bytes)

        for byteIndex in 0..<bytes {
            var byte: UInt8 = 0
            for bit in 0..<8 where vector[byteIndex * 8 + bit] > 0.5 {
                byte |= (1 << (7 - bit))
            }
            packed[byteIndex] = byte
        }
        return packed
    }
}
