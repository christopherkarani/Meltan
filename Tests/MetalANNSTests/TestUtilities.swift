import Foundation
import Metal

@testable import MetalANNSCore

// MARK: - Failing-Loud Environment Requirements

/// Throws `ANNSError.deviceNotSupported` instead of silently passing when required
/// test infrastructure is missing, so device-less runs FAIL rather than report success.
enum Require {
    /// Returns the system default Metal device or throws if none is available.
    static func metalDevice() throws -> any MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        return device
    }

    /// Returns a live `MetalContext` or throws if no Metal device is available.
    static func metalContext() throws -> MetalContext {
        try MetalContext()
    }

    /// Returns a live `MetalContext` or throws on simulator / device-less runs,
    /// logging `suite` so the skip reason names the suite being skipped.
    static func gpuContext(suite: String) throws -> MetalContext {
        #if targetEnvironment(simulator)
            print("Skipping \(suite) on simulator")
            throw ANNSError.deviceNotSupported
        #else
            try metalContext()
        #endif
    }
}

// MARK: - Temporary Directories

/// Creates a unique temporary directory, runs `body`, then removes the directory.
func withTempDirectory<T>(_ body: (URL) async throws -> T) async throws -> T {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("MetalANNS-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    return try await body(directory)
}

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
