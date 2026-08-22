import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Metal Distance Shader Tests")
struct MetalDistanceTests {
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

    @Test("GPU cosine matches CPU for 1000 random 128-dim vectors")
    func gpuVsCpuCosine() async throws {
        #if targetEnvironment(simulator)
            return
        #endif

        let dim = 128
        let n = 1000

        var vectors = [Float](repeating: 0, count: n * dim)
        fillWithSeededRandom(&vectors)
        var query = [Float](repeating: 0, count: dim)
        fillWithSeededRandom(&query, seed: 43)

        let cpuBackend = AccelerateBackend()
        let gpuBackend = try MetalBackend()

        let cpuResults = try await withVectorBuffer(vectors) { ptr in
            try await cpuBackend.computeDistances(
                query: query,
                vectors: ptr,
                vectorCount: n,
                dim: dim,
                metric: .cosine
            )
        }

        let gpuResults = try await withVectorBuffer(vectors) { ptr in
            try await gpuBackend.computeDistances(
                query: query,
                vectors: ptr,
                vectorCount: n,
                dim: dim,
                metric: .cosine
            )
        }

        #expect(cpuResults.count == gpuResults.count)
        for i in 0..<n {
            #expect(abs(cpuResults[i] - gpuResults[i]) < 1e-4)
        }
    }

    @Test("GPU L2 matches CPU for 1000 random 128-dim vectors")
    func gpuVsCpuL2() async throws {
        #if targetEnvironment(simulator)
            return
        #endif

        let dim = 128
        let n = 1000

        var vectors = [Float](repeating: 0, count: n * dim)
        fillWithSeededRandom(&vectors)
        var query = [Float](repeating: 0, count: dim)
        fillWithSeededRandom(&query, seed: 43)

        let cpuBackend = AccelerateBackend()
        let gpuBackend = try MetalBackend()

        let cpuResults = try await withVectorBuffer(vectors) { ptr in
            try await cpuBackend.computeDistances(
                query: query,
                vectors: ptr,
                vectorCount: n,
                dim: dim,
                metric: .l2
            )
        }

        let gpuResults = try await withVectorBuffer(vectors) { ptr in
            try await gpuBackend.computeDistances(
                query: query,
                vectors: ptr,
                vectorCount: n,
                dim: dim,
                metric: .l2
            )
        }

        #expect(cpuResults.count == gpuResults.count)
        for i in 0..<n {
            #expect(abs(cpuResults[i] - gpuResults[i]) < 1e-3)
        }
    }
}
