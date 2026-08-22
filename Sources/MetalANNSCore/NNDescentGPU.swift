import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "NNDescentGPU")

public enum NNDescentGPU {
    public static func randomInit(
        context: MetalContext,
        graph: GraphBuffer,
        nodeCount: Int,
        seed: UInt32 = 42
    ) async throws {
        guard nodeCount > 1, nodeCount <= graph.capacity else {
            throw ANNSError.constructionFailed("nodeCount must be at least 2 and within graph capacity")
        }

        let pipeline = try await context.pipelineCache.pipeline(for: "random_init")

        var n = UInt32(nodeCount)
        var degree = UInt32(graph.degree)
        var seedValue = seed

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBytes(&degree, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&seedValue, length: MemoryLayout<UInt32>.stride, index: 3)

            let threadsPerGrid = MTLSize(width: nodeCount, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(nodeCount, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        logger.debug("random_init complete for \(nodeCount) nodes")
    }

    public static func computeInitialDistances(
        context: MetalContext,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        nodeCount: Int,
        metric: Metric
    ) async throws {
        guard nodeCount > 1, nodeCount <= graph.capacity, nodeCount <= vectors.capacity else {
            throw ANNSError.constructionFailed("nodeCount must be at least 2 and within graph/vector capacity")
        }

        let kernelName = vectors.isFloat16 ? "compute_initial_distances_f16" : "compute_initial_distances"
        let pipeline = try await context.pipelineCache.pipeline(for: kernelName)

        var n = UInt32(nodeCount)
        var degree = UInt32(graph.degree)
        var dim = UInt32(vectors.dim)
        guard metric != .hamming else {
            throw ANNSError.constructionFailed("NNDescentGPU does not support metric .hamming")
        }
        var metricType: UInt32 =
            switch metric {
            case .cosine: 0
            case .l2: 1
            case .innerProduct: 2
            case .hamming: 0
            }

        let totalThreads = nodeCount * graph.degree

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 2)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&degree, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&metricType, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadsPerGrid = MTLSize(width: totalThreads, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(totalThreads, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
    }

    public static func build(
        context: MetalContext,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        nodeCount: Int,
        metric: Metric,
        maxIterations: Int = 20,
        convergenceThreshold: Float = 0.001
    ) async throws {
        guard nodeCount > 0, nodeCount <= graph.capacity, nodeCount <= vectors.capacity else {
            throw ANNSError.constructionFailed("nodeCount out of bounds for graph/vector capacity")
        }
        guard graph.degree <= 64 else {
            throw ANNSError.constructionFailed("Degree \(graph.degree) exceeds local_join forward array limit (64)")
        }

        let degree = graph.degree
        let maxReverse = degree * 2
        guard maxReverse <= 128 else {
            throw ANNSError.constructionFailed("maxReverse \(maxReverse) exceeds local_join reverse array limit (128)")
        }

        try await randomInit(context: context, graph: graph, nodeCount: nodeCount)
        try await computeInitialDistances(
            context: context,
            vectors: vectors,
            graph: graph,
            nodeCount: nodeCount,
            metric: metric
        )

        let reverseListLength = nodeCount * maxReverse * MemoryLayout<UInt32>.stride
        let reverseCountLength = nodeCount * MemoryLayout<UInt32>.stride
        let updateCountLength = MemoryLayout<UInt32>.stride

        guard
            let reverseListBuffer = context.device.makeBuffer(length: reverseListLength, options: .storageModeShared),
            let reverseCountBuffer = context.device.makeBuffer(length: reverseCountLength, options: .storageModeShared),
            let updateCountBuffer = context.device.makeBuffer(length: updateCountLength, options: .storageModeShared)
        else {
            throw ANNSError.constructionFailed("Failed to allocate NN-Descent GPU working buffers")
        }

        let reversePipeline = try await context.pipelineCache.pipeline(for: "build_reverse_list")
        let localJoinKernel = vectors.isFloat16 ? "local_join_f16" : "local_join"
        let localJoinPipeline = try await context.pipelineCache.pipeline(for: localJoinKernel)

        var n = UInt32(nodeCount)
        var d = UInt32(degree)
        var maxR = UInt32(maxReverse)
        var dim = UInt32(vectors.dim)
        guard metric != .hamming else {
            throw ANNSError.constructionFailed("NNDescentGPU does not support metric .hamming")
        }
        var metricType: UInt32 =
            switch metric {
            case .cosine: 0
            case .l2: 1
            case .innerProduct: 2
            case .hamming: 0
            }

        let reverseCounts = reverseCountBuffer.contents().bindMemory(to: UInt32.self, capacity: nodeCount)
        let updateCountPointer = updateCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)

        for iteration in 0..<maxIterations {
            for i in 0..<nodeCount {
                reverseCounts[i] = 0
            }
            updateCountPointer.pointee = 0

            let reverseThreads = nodeCount * degree
            try await context.execute { commandBuffer in
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw ANNSError.constructionFailed("Failed to create compute command encoder")
                }

                encoder.setComputePipelineState(reversePipeline)
                encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 0)
                encoder.setBuffer(reverseListBuffer, offset: 0, index: 1)
                encoder.setBuffer(reverseCountBuffer, offset: 0, index: 2)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 4)
                encoder.setBytes(&maxR, length: MemoryLayout<UInt32>.stride, index: 5)

                let threadsPerGrid = MTLSize(width: reverseThreads, height: 1, depth: 1)
                let threadgroupWidth = max(1, min(reverseThreads, reversePipeline.maxTotalThreadsPerThreadgroup))
                let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }

            try await context.execute { commandBuffer in
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw ANNSError.constructionFailed("Failed to create compute command encoder")
                }

                encoder.setComputePipelineState(localJoinPipeline)
                encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
                encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
                encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 2)
                encoder.setBuffer(reverseListBuffer, offset: 0, index: 3)
                encoder.setBuffer(reverseCountBuffer, offset: 0, index: 4)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 5)
                encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 6)
                encoder.setBytes(&maxR, length: MemoryLayout<UInt32>.stride, index: 7)
                encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 8)
                encoder.setBytes(&metricType, length: MemoryLayout<UInt32>.stride, index: 9)
                encoder.setBuffer(updateCountBuffer, offset: 0, index: 10)

                let threadsPerGrid = MTLSize(width: nodeCount, height: 1, depth: 1)
                let threadgroupWidth = max(1, min(nodeCount, localJoinPipeline.maxTotalThreadsPerThreadgroup))
                let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }

            let updateCount = updateCountPointer.pointee
            logger.debug("NNDescentGPU iteration \(iteration): \(updateCount) updates")

            if Float(updateCount) < convergenceThreshold * Float(degree * nodeCount) {
                logger.debug("NNDescentGPU converged at iteration \(iteration + 1)")
                break
            }
        }

        try await sortNeighborLists(context: context, graph: graph, nodeCount: nodeCount)
        graph.setCount(nodeCount)
    }

    public static func sortNeighborLists(
        context: MetalContext,
        graph: GraphBuffer,
        nodeCount: Int
    ) async throws {
        guard nodeCount > 0, nodeCount <= graph.capacity else {
            throw ANNSError.constructionFailed("nodeCount out of bounds for graph capacity")
        }

        let degree = graph.degree
        if degree <= 1 {
            return
        }
        guard degree <= 64 else {
            throw ANNSError.constructionFailed("Bitonic sort supports degree up to 64")
        }
        guard (degree & (degree - 1)) == 0 else {
            try sortNeighborListsCPU(graph: graph, nodeCount: nodeCount)
            return
        }

        let pipeline = try await context.pipelineCache.pipeline(for: "bitonic_sort_neighbors")
        let threadgroupWidth = degree / 2
        guard threadgroupWidth <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw ANNSError.constructionFailed("Degree \(degree) exceeds bitonic sort threadgroup capacity")
        }

        var degreeValue = UInt32(degree)

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 0)
            encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 1)
            encoder.setBytes(&degreeValue, length: MemoryLayout<UInt32>.stride, index: 2)

            let threadgroups = MTLSize(width: nodeCount, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
    }

    private static func sortNeighborListsCPU(graph: GraphBuffer, nodeCount: Int) throws {
        for nodeID in 0..<nodeCount {
            let ids = graph.neighborIDs(of: nodeID)
            let distances = graph.neighborDistances(of: nodeID)
            var pairs = Array(zip(ids, distances))
            pairs.sort { lhs, rhs in
                if lhs.1 == rhs.1 {
                    return lhs.0 < rhs.0
                }
                return lhs.1 < rhs.1
            }
            try graph.setNeighbors(
                of: nodeID,
                ids: pairs.map(\.0),
                distances: pairs.map(\.1)
            )
        }
    }
}
