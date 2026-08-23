import Foundation
import Metal
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Typed Error Model Tests")
struct ErrorModelTests {
    @Test("CancellationError surfaces verbatim from HNSWSearchCPU")
    func cancellationSurfacesVerbatim() async throws {
        let vectors = seededVectors(count: 8000, dim: 8, seed: 7)
        let graph = ringGraph(count: vectors.count)
        let query = seededVector(dim: 8, seed: 9)

        var observedCancellation = false
        for _ in 0..<3 {
            let task = Task {
                try await HNSWSearchCPU.search(
                    query: query,
                    vectors: vectors,
                    hnsw: HNSWLayers(),
                    baseGraph: graph,
                    k: 8,
                    ef: 32,
                    metric: .l2
                )
            }
            task.cancel()
            do {
                _ = try await task.value
                continue
            } catch {
                #expect(error is CancellationError)
                observedCancellation = true
                break
            }
        }
        #expect(observedCancellation)
    }

    @Test("Underlying beam errors propagate unchanged through HNSWSearchCPU")
    func underlyingBeamErrorsPropagateUnchanged() async throws {
        let vectors = seededVectors(count: 4, dim: 8, seed: 3)
        do {
            _ = try await HNSWSearchCPU.search(
                query: seededVector(dim: 8, seed: 5),
                vectors: vectors,
                hnsw: HNSWLayers(),
                baseGraph: ringGraph(count: vectors.count),
                k: 4,
                ef: 2,
                metric: .l2
            )
            Issue.record("Expected ef < k to throw from the beam tier")
        } catch let error as ANNSError {
            guard case .searchFailed(let message) = error else {
                Issue.record("Unexpected ANNSError case: \(error)")
                return
            }
            #expect(message == "ef must be greater than or equal to k")
        }
    }

    @Test("GraphIndex insert overflow throws indexCapacityExceeded")
    func capacityCaseThrownAtSource() async throws {
        let index = GraphIndex(configuration: IndexConfiguration(degree: 2, metric: .l2))
        let baseVectors = seededVectors(count: 4, dim: 8, seed: 11)
        try await index.build(vectors: baseVectors, ids: (0..<baseVectors.count).map { "v\($0)" })

        var overflowCount = 0
        for i in 0..<6 {
            do {
                try await index.insert(seededVector(dim: 8, seed: UInt64(100 + i)), id: "overflow-\(i)")
            } catch let error as ANNSError {
                guard case .indexCapacityExceeded = error else {
                    Issue.record("Unexpected ANNSError case on insert \(i): \(error)")
                    return
                }
                overflowCount += 1
            }
        }

        #expect(
            overflowCount == 2,
            "Build reserves 2x headroom: 4 inserts fit, the rest must throw typed capacity errors"
        )

        await expectCase(.indexCapacityExceeded) {
            try await index.batchInsert([seededVector(dim: 8, seed: 13)], ids: ["batched-overflow"])
        }
    }

    @Test("Typed capacity case triggers streaming merge end to end")
    func typedCapacityCaseTriggersMerge() async throws {
        let deltaCapacity = 4
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: deltaCapacity,
                mergeStrategy: .blocking
            ))

        let total = deltaCapacity * 6
        let vectors = (0..<total).map { makeVector(row: $0, dim: 8) }
        for (i, vector) in vectors.enumerated() {
            try await index.insert(vector, id: "v\(i)")
        }

        #expect(await index.count == total)
        #expect(await index.isMerging == false)
        for (i, vector) in vectors.enumerated() {
            let results = try await index.search(query: vector, k: 1)
            #expect(!results.isEmpty)
            #expect(results[0].id == "v\(i)")
        }
    }

    @Test("Prose-based error matching no longer exists in Sources")
    func proseMatchingRemovedFromSources() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourcesURL = repoRoot.appendingPathComponent("Sources")

        guard let enumerator = FileManager.default.enumerator(at: sourcesURL, includingPropertiesForKeys: nil) else {
            Issue.record("Could not enumerate Sources at \(sourcesURL.path)")
            return
        }

        var bannedMatches: [String] = []
        for case let fileURL as URL in enumerator where fileURL.pathExtension == "swift" {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            if contents.contains("contains(\"Index capacity exceeded\")") {
                bannedMatches.append(fileURL.path)
            }
            if contents.contains("HNSW layer-0 beam search failed") {
                bannedMatches.append(fileURL.path)
            }
        }
        #expect(bannedMatches.isEmpty, "Prose matching survived in: \(bannedMatches)")
    }

    @Test("Missing shader function surfaces as gpuPipelineUnavailable")
    func missingShaderFunctionThrowsTypedCase() async throws {
        guard let context = makeGPUContextOrSkip() else {
            return
        }
        let cache = PipelineCache(device: context.device, library: context.library)
        do {
            _ = try await cache.pipeline(for: "definitely_missing_shader_0xF00D")
            Issue.record("Expected gpuPipelineUnavailable for unknown function name")
        } catch let error as ANNSError {
            guard case .gpuPipelineUnavailable(let message) = error else {
                Issue.record("Unexpected ANNSError case: \(error)")
                return
            }
            #expect(message.contains("definitely_missing_shader_0xF00D"))
        }
    }

    @Test("Additive typed cases carry their payloads")
    func additiveCasesCarryPayloads() {
        guard case .gpuPipelineUnavailable(let pipelineMessage) = ANNSError.gpuPipelineUnavailable("encoder") else {
            Issue.record("gpuPipelineUnavailable payload lost")
            return
        }
        #expect(pipelineMessage == "encoder")

        guard case .gpuResourceExhausted(let resourceMessage) = ANNSError.gpuResourceExhausted("workspace") else {
            Issue.record("gpuResourceExhausted payload lost")
            return
        }
        #expect(resourceMessage == "workspace")

        guard case .gpuExecutionFailed(let executionMessage) = ANNSError.gpuExecutionFailed("command buffer") else {
            Issue.record("gpuExecutionFailed payload lost")
            return
        }
        #expect(executionMessage == "command buffer")

        #expect(isCaseEqual(ANNSError.indexCapacityExceeded, ANNSError.indexCapacityExceeded))
    }

    private func expectCase(
        _ expected: ANNSError,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            Issue.record("Expected \(expected), but no error was thrown")
        } catch let error as ANNSError {
            guard isCaseEqual(error, expected) else {
                Issue.record("Expected \(expected), got \(error)")
                return
            }
        } catch {
            Issue.record("Expected \(expected), got non-ANNSError: \(error)")
        }
    }

    private func isCaseEqual(_ lhs: ANNSError, _ rhs: ANNSError) -> Bool {
        switch (lhs, rhs) {
        case (.indexCapacityExceeded, .indexCapacityExceeded):
            return true
        case (.deviceNotSupported, .deviceNotSupported):
            return true
        default:
            return false
        }
    }

    private func makeGPUContextOrSkip() -> MetalContext? {
        #if targetEnvironment(simulator)
            print("Skipping ErrorModelTests on simulator")
            return nil
        #else
            guard MTLCreateSystemDefaultDevice() != nil else {
                print("Skipping ErrorModelTests: no Metal device available")
                return nil
            }
            do {
                return try MetalContext()
            } catch {
                print("Skipping ErrorModelTests: MetalContext init failed: \(error)")
                return nil
            }
        #endif
    }

    private func ringGraph(count: Int) -> [[(UInt32, Float)]] {
        guard count > 1 else {
            return [[(UInt32.max, 0)]]
        }
        var graph: [[(UInt32, Float)]] = []
        graph.reserveCapacity(count)
        for i in 0..<count {
            let next = (i + 1) % count
            let previous = (i + count - 1) % count
            let neighbors: [(UInt32, Float)] = [
                (UInt32(next), 1.0),
                (UInt32(previous), 1.0),
            ]
            graph.append(neighbors)
        }
        return graph
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.091) + cos(i * 0.037)
        }
    }
}
