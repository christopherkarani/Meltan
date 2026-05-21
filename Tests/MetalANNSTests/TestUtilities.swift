import Foundation

// MARK: - Deterministic Random Generation

/// Deterministic XOR-shift PRNG for reproducible test data.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    init(state: UInt64) {
        self.state = state == 0 ? 1 : state
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

/// Generates `count` random vectors of dimension `dim` using a fixed seed.
/// All test data should use this helper to ensure reproducible failures.
func seededVectors(count: Int, dim: Int, seed: UInt64 = 42) -> [[Float]] {
    var rng = SeededGenerator(state: seed)
    return (0..<count).map { _ in
        (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    }
}

/// Generates a single random vector of dimension `dim` using a fixed seed.
func seededVector(dim: Int, seed: UInt64 = 42) -> [Float] {
    var rng = SeededGenerator(state: seed)
    return (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
}

/// Fills an existing `UnsafeMutablePointer<Float>` or `[Float]` buffer with seeded random values.
func fillWithSeededRandom(_ buffer: inout [Float], seed: UInt64 = 42) {
    var rng = SeededGenerator(state: seed)
    for i in buffer.indices {
        buffer[i] = Float.random(in: -1...1, using: &rng)
    }
}

/// Generates `count` random vectors clustered around `centers` for IVF/PQ tests.
func seededClusteredVectors(count: Int, dim: Int, centers: [[Float]], seed: UInt64 = 42) -> [[Float]] {
    var rng = SeededGenerator(state: seed)
    return (0..<count).map { i in
        let center = centers[i % centers.count]
        return center.enumerated().map { _, c in
            c + Float.random(in: -0.02...0.02, using: &rng)
        }
    }
}
