import Foundation
import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("GPU ADC Search Tests")
struct GPUADCSearchTests {
    @Test("GPU ADC distances match CPU ADC distances (tolerance 1e-3)")
    func gpuDistancesMatchCPU() async throws {
        let context = try requireGPUContext()
        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 200, dim: 64, seed: 302)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 303)[0]

        let gpu = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes
        )

        #expect(gpu.count == codes.count)
        for i in 0..<codes.count {
            let cpu = pq.approximateDistance(query: query, codes: codes[i], metric: .l2)
            #expect(abs(gpu[i] - cpu) < 1e-3, "Difference at \(i): cpu=\(cpu) gpu=\(gpu[i])")
        }
    }

    @Test("Empty code input returns empty distance output")
    func emptyCodesReturnsEmpty() async throws {
        let context = try requireGPUContext()
        let pq = try trainPQ(dim: 64, M: 8)
        let query = makeRandomVectors(count: 1, dim: 64, seed: 304)[0]

        let distances = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: []
        )

        #expect(distances.isEmpty)
    }

    @Test("Flatten codebooks uses M x Ks x subspaceDim layout")
    func flattenCodebooksCorrectLayout() throws {
        let pq = try trainPQ(dim: 16, M: 4)
        let flattened = GPUADCSearch.flattenCodebooks(from: pq)
        let expectedCount = pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension

        #expect(flattened.count == expectedCount)
        #expect(Array(flattened.prefix(pq.subspaceDimension)) == pq.codebooks[0][0])
    }

    @Test("Providing cached flat codebooks matches nil flatCodebooks path")
    func cachedFlatCodebooksSkipsRecomputation() async throws {
        let context = try requireGPUContext()
        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 200, dim: 64, seed: 305)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 306)[0]
        let cachedFlat = GPUADCSearch.flattenCodebooks(from: pq)

        let cached = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            flatCodebooks: cachedFlat
        )

        let recomputed = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            flatCodebooks: nil
        )

        #expect(cached.count == recomputed.count)
        for i in 0..<cached.count {
            #expect(abs(cached[i] - recomputed[i]) < 1e-6)
        }
    }

    @Test("IVFPQ search parity after GPUADCSearch delegation")
    func ivfpqRegressionAfterRewire() async throws {
        _ = try requireGPUContext()

        let config = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 6
        )
        let index = try Advanced.IVFPQIndex(capacity: 2_500, dimension: 128, config: config)

        let centers = makeClusterCenters(dim: 128, clusters: 32, seed: 401)
        let training = sampleClusteredVectors(count: 1_000, centers: centers, seed: 402)
        try await index.train(vectors: training)

        let database = sampleClusteredVectors(count: 600, centers: centers, seed: 403)
        let ids = (0..<database.count).map { "ivfpq-\($0)" }
        try await index.add(vectors: database, ids: ids)

        let query = sampleClusteredVectors(count: 1, centers: centers, seed: 404)[0]
        let gpu = await index.search(query: query, k: 10, nprobe: 8, forceGPU: true)
        let cpu = await index.search(query: query, k: 10, nprobe: 8, forceGPU: false)

        #expect(gpu.count == 10)
        #expect(cpu.count == 10)
        #expect(Set(gpu.map(\.id)) == Set(cpu.map(\.id)))
    }

    @Test("search returns sorted top-k and top-1 matches CPU brute-force ADC")
    func searchReturnsTopK() async throws {
        let context = try requireGPUContext()
        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 300, dim: 64, seed: 501)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let ids = (0..<codes.count).map { "gpuadc-\($0)" }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 502)[0]

        let results = try await GPUADCSearch.search(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            ids: ids,
            k: 10
        )

        #expect(results.count == 10)
        for i in 1..<results.count {
            #expect(results[i].score >= results[i - 1].score)
        }

        let cpuTop = zip(ids, codes).map { id, code in
            (id, pq.approximateDistance(query: query, codes: code, metric: .l2))
        }
        .sorted { $0.1 < $1.1 }
        .first

        #expect(results.first?.id == cpuTop?.0)
    }

    @Test("search returns all results when k exceeds corpus size")
    func searchKLargerThanCorpus() async throws {
        let context = try requireGPUContext()
        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 5, dim: 64, seed: 503)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let ids = (0..<codes.count).map { "small-\($0)" }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 504)[0]

        let results = try await GPUADCSearch.search(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            ids: ids,
            k: 100
        )

        #expect(results.count == 5)
        for i in 1..<results.count {
            #expect(results[i].score >= results[i - 1].score)
        }
    }

    @Test("small corpus GPU ADC remains correct")
    func gpuDistancesMatchCPUSmallCorpus() async throws {
        let context = try requireGPUContext()
        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 5, dim: 64, seed: 701)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 702)[0]

        let gpu = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes
        )

        #expect(gpu.count == codes.count)
        for i in 0..<codes.count {
            let cpu = pq.approximateDistance(query: query, codes: codes[i], metric: .l2)
            #expect(abs(gpu[i] - cpu) < 1e-3, "Difference at \(i): cpu=\(cpu) gpu=\(gpu[i])")
        }
    }

    @Test("rankDistances sorts and truncates without GPU")
    func rankDistancesSortAndTopK() throws {
        let distances: [Float] = [3.0, 1.0, 2.0, 0.5]
        let ids = ["d", "b", "c", "a"]

        let results = try GPUADCSearch.rankDistances(
            distances: distances,
            ids: ids,
            k: 3
        )

        #expect(results.count == 3)
        #expect(results.map(\.id) == ["a", "b", "c"])
        #expect(results.map(\.internalID) == [3, 1, 2])
    }

    @Test("rankDistances validates input and k edge cases without GPU")
    func rankDistancesValidationAndKEdges() throws {
        #expect(throws: ANNSError.self) {
            _ = try GPUADCSearch.rankDistances(
                distances: [1.0, 2.0],
                ids: ["only-one"],
                k: 1
            )
        }

        let emptyForZeroK = try GPUADCSearch.rankDistances(
            distances: [1.0, 2.0],
            ids: ["a", "b"],
            k: 0
        )
        #expect(emptyForZeroK.isEmpty)

        let emptyForNoDistances = try GPUADCSearch.rankDistances(
            distances: [],
            ids: [],
            k: 5
        )
        #expect(emptyForNoDistances.isEmpty)
    }

    @Test("GPU ADC rejects out-of-range PQ code values")
    func rejectsOutOfRangeCodeValues() async throws {
        let context = try requireGPUContext()
        let pq = ProductQuantizer(
            numSubspaces: 2,
            centroidsPerSubspace: 4,
            subspaceDimension: 2,
            codebooks: [
                [[0, 0], [1, 1], [2, 2], [3, 3]],
                [[0, 0], [1, 1], [2, 2], [3, 3]],
            ]
        )

        do {
            _ = try await GPUADCSearch.computeDistances(
                context: context,
                query: [0, 0, 0, 0],
                pq: pq,
                codes: [[0, 250]]
            )
            #expect(Bool(false), "Expected out-of-range code value rejection")
        } catch let error as ANNSError {
            guard case .searchFailed = error else {
                #expect(Bool(false), "Expected searchFailed, got \(error)")
                return
            }
        }
    }

    @Test("GPUADCSearch rejects table that exceeds device threadgroup memory")
    func rejectsDistanceTableExceedingThreadgroupLimit() async throws {
        let context = try requireGPUContext()
        let maxThreadgroupBytes = context.device.maxThreadgroupMemoryLength
        let m = 256
        let ks = 256
        let tableLengthBytes = m * ks * MemoryLayout<Float>.stride

        guard tableLengthBytes > maxThreadgroupBytes else {
            print("Skipping oversized-table guard test: device threadgroup limit is unexpectedly large")
            return
        }

        let pq = ProductQuantizer(
            numSubspaces: m,
            centroidsPerSubspace: ks,
            subspaceDimension: 1,
            codebooks: Array(
                repeating: Array(repeating: [Float(0)], count: ks),
                count: m
            )
        )
        let query = Array(repeating: Float(0), count: m)
        let codes = Array(repeating: Array(repeating: UInt8(0), count: m), count: 4)

        do {
            _ = try await GPUADCSearch.computeDistances(
                context: context,
                query: query,
                pq: pq,
                codes: codes
            )
            #expect(Bool(false), "Expected oversized threadgroup-table rejection")
        } catch let error as ANNSError {
            guard case .searchFailed(let message) = error else {
                #expect(Bool(false), "Expected searchFailed for oversized threadgroup table, got \(error)")
                return
            }
            #expect(
                message.contains("threadgroup") || message.contains("table"),
                "Error message should mention threadgroup/table sizing, got: \(message)"
            )
        }
    }

    private func requireGPUContext() throws -> MetalContext {
        #if targetEnvironment(simulator)
            print("Skipping GPU ADC tests on simulator")
            throw ANNSError.deviceNotSupported
        #else
            try Require.metalContext()
        #endif
    }
}

// swift-format-ignore: AlwaysUseLowerCamelCase
private func trainPQ(dim: Int = 64, M: Int = 8) throws -> ProductQuantizer {
    let vectors = makeRandomVectors(count: 500, dim: dim, seed: 301)
    return try ProductQuantizer.train(
        vectors: vectors,
        numSubspaces: M,
        centroidsPerSubspace: 256,
        maxIterations: 6
    )
}

private func makeRandomVectors(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<count).map { _ in
        (0..<dim).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
    }
}

private func makeClusterCenters(dim: Int, clusters: Int, seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<clusters).map { _ in
        (0..<dim).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
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
