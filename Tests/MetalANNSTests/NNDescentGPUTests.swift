import Metal
import Testing
@testable import MetalANNSCore

@Suite("GPU NN-Descent Tests")
struct NNDescentGPUTests {
    private let localJoinNodeCount = 3
    private let localJoinDegree = 2
    private let localJoinDim = 2
    private let localJoinMaxReverse = 4

    private func withVectorBuffer<T>(
        _ values: [Float],
        _ body: (UnsafeBufferPointer<Float>) async throws -> T
    ) async throws -> T {
        let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
        _ = buffer.initialize(from: values)
        defer {
            buffer.deinitialize()
            buffer.deallocate()
        }
        return try await body(UnsafeBufferPointer(buffer))
    }

    private func makeLocalJoinGraph(context: MetalContext) throws -> GraphBuffer {
        let graph = try GraphBuffer(
            capacity: localJoinNodeCount,
            degree: localJoinDegree,
            device: context.device
        )

        try graph.setNeighbors(
            of: 0,
            ids: [1, UInt32.max],
            distances: [1.0, Float.greatestFiniteMagnitude]
        )
        try graph.setNeighbors(
            of: 1,
            ids: [0, UInt32.max],
            distances: [100.0, Float.greatestFiniteMagnitude]
        )
        try graph.setNeighbors(
            of: 2,
            ids: [0, UInt32.max],
            distances: [100.0, Float.greatestFiniteMagnitude]
        )
        graph.setCount(localJoinNodeCount)

        return graph
    }

    private func makeReverseBuffers(
        context: MetalContext
    ) throws -> (reverseList: MTLBuffer, reverseCounts: MTLBuffer) {
        let reverseListLength = localJoinNodeCount * localJoinMaxReverse * MemoryLayout<UInt32>.stride
        let reverseCountLength = localJoinNodeCount * MemoryLayout<UInt32>.stride

        guard
            let reverseListBuffer = context.device.makeBuffer(length: reverseListLength, options: .storageModeShared),
            let reverseCountBuffer = context.device.makeBuffer(length: reverseCountLength, options: .storageModeShared)
        else {
            throw ANNSError.constructionFailed("Failed to allocate local_join reverse buffers")
        }

        let reverseList = reverseListBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: localJoinNodeCount * localJoinMaxReverse
        )
        for idx in 0..<(localJoinNodeCount * localJoinMaxReverse) {
            reverseList[idx] = UInt32.max
        }

        // Thread tid=0 uses fwd={1} and rev={1,2}; only pair (1,2) should be refined.
        reverseList[0] = 1
        reverseList[1] = 2

        let reverseCounts = reverseCountBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: localJoinNodeCount
        )
        for idx in 0..<localJoinNodeCount {
            reverseCounts[idx] = 0
        }
        reverseCounts[0] = 2

        return (reverseListBuffer, reverseCountBuffer)
    }

    private func runLocalJoinPass(
        context: MetalContext,
        kernelName: String,
        vectorsBuffer: MTLBuffer,
        graph: GraphBuffer,
        reverseListBuffer: MTLBuffer,
        reverseCountBuffer: MTLBuffer
    ) async throws -> UInt32 {
        let localJoinPipeline = try await context.pipelineCache.pipeline(for: kernelName)
        guard let updateCountBuffer = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw ANNSError.constructionFailed("Failed to allocate local_join update counter")
        }
        let updateCounter = updateCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        updateCounter[0] = 0

        var nodeCount = UInt32(localJoinNodeCount)
        var degree = UInt32(localJoinDegree)
        var maxReverse = UInt32(localJoinMaxReverse)
        var dim = UInt32(localJoinDim)
        var metricType: UInt32 = 1 // L2

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute encoder for local_join regression test")
            }

            encoder.setComputePipelineState(localJoinPipeline)
            encoder.setBuffer(vectorsBuffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 2)
            encoder.setBuffer(reverseListBuffer, offset: 0, index: 3)
            encoder.setBuffer(reverseCountBuffer, offset: 0, index: 4)
            encoder.setBytes(&nodeCount, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&degree, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&maxReverse, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 8)
            encoder.setBytes(&metricType, length: MemoryLayout<UInt32>.stride, index: 9)
            encoder.setBuffer(updateCountBuffer, offset: 0, index: 10)

            let threadsPerGrid = MTLSize(width: localJoinNodeCount, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(localJoinNodeCount, localJoinPipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        return updateCounter[0]
    }

    @Test("Random init produces valid graph")
    func randomInitValid() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let context = try MetalContext()
        let nodeCount = 100
        let degree = 8
        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)

        try await NNDescentGPU.randomInit(
            context: context,
            graph: graph,
            nodeCount: nodeCount,
            seed: 42
        )

        for node in 0..<nodeCount {
            let neighborIDs = graph.neighborIDs(of: node)
            #expect(neighborIDs.count == degree)
            #expect(Set(neighborIDs).count == degree, "Duplicate neighbors at node \(node)")

            for neighborID in neighborIDs {
                #expect(neighborID != UInt32(node), "Self-loop at node \(node)")
                #expect(neighborID < UInt32(nodeCount), "Out-of-range neighbor \(neighborID) at node \(node)")
            }
        }
    }

    @Test("Full GPU NN-Descent construction recall > 0.80")
    func fullGPUConstruction() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 200
        let dim = 16
        let degree = 8
        let maxIterations = 15

        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1) }
        }

        let context = try MetalContext()
        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: context.device)
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)

        try await NNDescentGPU.build(
            context: context,
            vectors: vectorBuffer,
            graph: graph,
            nodeCount: nodeCount,
            metric: .cosine,
            maxIterations: maxIterations
        )

        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        var totalRecall: Float = 0

        for node in 0..<nodeCount {
            let distances = try await withVectorBuffer(flat) { pointer in
                try await backend.computeDistances(
                    query: vectors[node],
                    vectors: pointer,
                    vectorCount: nodeCount,
                    dim: dim,
                    metric: .cosine
                )
            }

            let exactTopK = Set(
                distances.enumerated()
                    .filter { $0.offset != node }
                    .sorted { $0.element < $1.element }
                    .prefix(degree)
                    .map { UInt32($0.offset) }
            )

            let graphTopK = Set(
                graph.neighborIDs(of: node).filter { $0 != UInt32.max && $0 != UInt32(node) && $0 < UInt32(nodeCount) }
            )

            let overlap = exactTopK.intersection(graphTopK).count
            totalRecall += Float(overlap) / Float(degree)
        }

        let averageRecall = totalRecall / Float(nodeCount)
        #expect(averageRecall > 0.80, "Average recall \\(averageRecall) below 0.80")
    }

    @Test("Random init rejects nodeCount less than 2")
    func randomInitRejectsSingleNode() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let context = try MetalContext()
        let graph = try GraphBuffer(capacity: 1, degree: 1, device: context.device)

        do {
            try await NNDescentGPU.randomInit(
                context: context,
                graph: graph,
                nodeCount: 1
            )
            #expect(Bool(false), "Expected randomInit to reject nodeCount < 2")
        } catch let error as ANNSError {
            guard case .constructionFailed = error else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
                return
            }
        }
    }

    @Test("local_join updates candidate pairs (float32)")
    func localJoinUpdatesPairwiseFloat32() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let context = try MetalContext()
        let graph = try makeLocalJoinGraph(context: context)
        let (reverseListBuffer, reverseCountBuffer) = try makeReverseBuffers(context: context)

        let vectors = try VectorBuffer(capacity: localJoinNodeCount, dim: localJoinDim, device: context.device)
        try vectors.batchInsert(
            vectors: [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.2, 0.0]
            ],
            startingAt: 0
        )
        vectors.setCount(localJoinNodeCount)

        let updates = try await runLocalJoinPass(
            context: context,
            kernelName: "local_join",
            vectorsBuffer: vectors.buffer,
            graph: graph,
            reverseListBuffer: reverseListBuffer,
            reverseCountBuffer: reverseCountBuffer
        )

        let node1 = graph.neighborIDs(of: 1)
        let node2 = graph.neighborIDs(of: 2)

        #expect(updates >= 2, "Expected pairwise refinement to produce at least two updates, got \(updates)")
        #expect(node1.contains(2), "Expected node 1 to add node 2 after local_join pairwise refinement")
        #expect(node2.contains(1), "Expected node 2 to add node 1 after local_join pairwise refinement")
    }

    @Test("local_join_f16 updates candidate pairs (float16)")
    func localJoinUpdatesPairwiseFloat16() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let context = try MetalContext()
        let graph = try makeLocalJoinGraph(context: context)
        let (reverseListBuffer, reverseCountBuffer) = try makeReverseBuffers(context: context)

        let vectors = try Float16VectorBuffer(capacity: localJoinNodeCount, dim: localJoinDim, device: context.device)
        try vectors.batchInsert(
            vectors: [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.2, 0.0]
            ],
            startingAt: 0
        )
        vectors.setCount(localJoinNodeCount)

        let updates = try await runLocalJoinPass(
            context: context,
            kernelName: "local_join_f16",
            vectorsBuffer: vectors.buffer,
            graph: graph,
            reverseListBuffer: reverseListBuffer,
            reverseCountBuffer: reverseCountBuffer
        )

        let node1 = graph.neighborIDs(of: 1)
        let node2 = graph.neighborIDs(of: 2)

        #expect(updates >= 2, "Expected pairwise refinement to produce at least two updates, got \(updates)")
        #expect(node1.contains(2), "Expected node 1 to add node 2 after local_join_f16 pairwise refinement")
        #expect(node2.contains(1), "Expected node 2 to add node 1 after local_join_f16 pairwise refinement")
    }
}
