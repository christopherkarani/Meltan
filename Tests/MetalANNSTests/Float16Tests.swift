import Metal
import MetalANNS
import MetalANNSCore
import Testing

@Suite("Float16 Buffer and Distance")
struct Float16Tests {
    @Test func float16DistanceMatchesFloat32() async throws {
        let dim = 64
        let n = 50
        let vectors = (0..<n).map { i in
            seededVector(dim: dim, seed: 300 + UInt64(i))
        }

        let device = MTLCreateSystemDefaultDevice()

        let f32Buffer = try VectorBuffer(capacity: n, dim: dim, device: device)
        try f32Buffer.batchInsert(vectors: vectors, startingAt: 0)
        f32Buffer.setCount(n)

        let f16Buffer = try Float16VectorBuffer(capacity: n, dim: dim, device: device)
        try f16Buffer.batchInsert(vectors: vectors, startingAt: 0)
        f16Buffer.setCount(n)

        for i in 0..<n {
            let f32Vec = f32Buffer.vector(at: i)
            let f16Vec = f16Buffer.vector(at: i)
            #expect(f32Vec.count == f16Vec.count)

            for d in 0..<dim {
                let diff = abs(f32Vec[d] - f16Vec[d])
                #expect(diff < 0.01)
            }
        }

        let query = seededVector(dim: dim, seed: 777)
        for i in 0..<min(10, n) {
            let f32Dist = SIMDDistance.cosine(query, f32Buffer.vector(at: i))
            let f16Dist = SIMDDistance.cosine(query, f16Buffer.vector(at: i))
            let relativeError = abs(f32Dist - f16Dist) / max(abs(f32Dist), 1e-10)
            #expect(relativeError < 0.05)
        }
    }

    @Test func float16RecallComparable() async throws {
        let dim = 32
        let n = 200
        let k = 10
        let vectors = (0..<n).map { i in
            seededVector(dim: dim, seed: 400 + UInt64(i))
        }
        let ids = (0..<n).map { "v_\($0)" }

        let f32Config = IndexConfiguration(degree: 8, metric: .cosine, useFloat16: false)
        let f32Index = GraphIndex(configuration: f32Config)
        try await f32Index.build(vectors: vectors, ids: ids)

        let f16Config = IndexConfiguration(degree: 8, metric: .cosine, useFloat16: true)
        let f16Index = GraphIndex(configuration: f16Config)
        try await f16Index.build(vectors: vectors, ids: ids)

        let queries = (0..<5).map { i in
            seededVector(dim: dim, seed: 5000 + UInt64(i))
        }

        var totalOverlap = 0
        let totalExpected = queries.count * k

        for query in queries {
            let f32Results = try await f32Index.search(query: query, k: k)
            let f16Results = try await f16Index.search(query: query, k: k)

            let f32IDs = Set(f32Results.map(\.id))
            let f16IDs = Set(f16Results.map(\.id))
            totalOverlap += f32IDs.intersection(f16IDs).count
        }

        let recall = Float(totalOverlap) / Float(totalExpected)
        #expect(recall >= 0.5)
    }
}
