import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("IVFPQ GPU Tests")
struct IVFPQGPUTests {
    @Test("GPU ADC distances match CPU ADC distances (tolerance 1e-3)")
    func gpuVsCpuDistances() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            let config = IVFPQConfiguration(
                numSubspaces: 8,
                numCentroids: 256,
                numCoarseCentroids: 32,
                nprobe: 8,
                metric: .l2,
                trainingIterations: 6
            )
            let index = try Advanced.IVFPQIndex(capacity: 2_500, dimension: 128, config: config)

            guard await index.gpuAvailable() else {
                return
            }

            let centers = makeClusterCenters(dimension: 128, clusters: 32, seed: 201)
            let training = sampleClusteredVectors(count: 1_000, centers: centers, seed: 202)
            try await index.train(vectors: training)

            let database = sampleClusteredVectors(count: 600, centers: centers, seed: 203)
            let ids = (0..<database.count).map { "gpu-\($0)" }
            try await index.add(vectors: database, ids: ids)

            guard let cluster = await index.firstNonEmptyCluster() else {
                Issue.record("No non-empty cluster found for GPU test")
                return
            }

            let query = sampleClusteredVectors(count: 1, centers: centers, seed: 204)[0]
            let cpu = try await index.debugClusterDistances(query: query, clusterIndex: cluster, forceGPU: false)
            let gpu = try await index.debugClusterDistances(query: query, clusterIndex: cluster, forceGPU: true)

            #expect(cpu.count == gpu.count)
            #expect(!cpu.isEmpty)
            for i in 0..<cpu.count {
                #expect(abs(cpu[i] - gpu[i]) < 1e-3, "Difference at \(i): cpu=\(cpu[i]) gpu=\(gpu[i])")
            }
        #endif
    }
}

private func makeClusterCenters(dimension: Int, clusters: Int, seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<clusters).map { _ in
        (0..<dimension).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
    }
}

private func sampleClusteredVectors(count: Int, centers: [[Float]], seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    let clusters = centers.count
    return (0..<count).map { row in
        let cluster = row % clusters
        return centers[cluster].map { center in
            center + Float.random(in: -0.02...0.02, using: &rng)
        }
    }
}
