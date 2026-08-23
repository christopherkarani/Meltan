import Foundation
import Metal

package enum SearchGPU {
    // FIXME: Major — per-iteration GPU dispatch is catastrophically slow.
    // Each beam-search iteration dispatches one command buffer + allocates buffers.
    // For ef=200 iterating 100+ times, pure dispatch overhead reaches 5–20 ms.
    // Should batch all neighbor distances into a single dispatch or use a fully
    // GPU-resident kernel.
    private static let workspacePool = WorkspaceBufferPool<Workspace>(
        maxRetainedEntries: 16,
        entryBytes: { $0.neighborBuffer.length + $0.outputBuffer.length }
    )

    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    package static func search(
        context: MetalContext,
        query: [Float],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        let nodeCount = graph.nodeCount > 0 ? min(graph.nodeCount, vectors.count) : min(graph.capacity, vectors.count)

        guard nodeCount > 0 else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }
        guard ef >= k else {
            throw ANNSError.searchFailed("ef must be greater than or equal to k")
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard entryPoint >= 0, entryPoint < nodeCount else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        let efLimit = min(ef, nodeCount)
        let entryID = UInt32(entryPoint)
        let queryBuffer = try makeQueryBuffer(
            context: context,
            query: query,
            vectors: vectors
        )
        let entryDistance = SIMDDistance.distance(
            query,
            vectors.vector(at: entryPoint),
            metric: metric
        )

        var visited: Set<UInt32> = [entryID]
        var candidates = BinaryHeap<Candidate> { lhs, rhs in
            lhs.distance < rhs.distance
        }
        var results: [Candidate] = [Candidate(nodeID: entryID, distance: entryDistance)]
        candidates.push(Candidate(nodeID: entryID, distance: entryDistance))

        while let current = candidates.pop() {
            if results.count >= efLimit, let worst = results.last, current.distance > worst.distance {
                break
            }

            let rawNeighbors = graph.neighborIDs(of: Int(current.nodeID))
            var neighborIDs: [UInt32] = []
            neighborIDs.reserveCapacity(rawNeighbors.count)

            for neighborID in rawNeighbors {
                let neighborIndex = Int(neighborID)
                if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= nodeCount {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)
                neighborIDs.append(neighborID)
            }

            if neighborIDs.isEmpty {
                continue
            }

            let neighborDistances = try await computeDistancesOnGPU(
                context: context,
                queryBuffer: queryBuffer,
                vectors: vectors,
                neighborIDs: neighborIDs,
                metric: metric
            )

            for (offset, distanceValue) in neighborDistances.enumerated() {
                if results.count < efLimit || distanceValue < results[results.count - 1].distance {
                    let candidate = Candidate(nodeID: neighborIDs[offset], distance: distanceValue)
                    candidates.push(candidate)
                    insertSorted(candidate, into: &results)
                    if results.count > efLimit {
                        results.removeLast()
                    }
                }
            }
        }

        let topK = min(k, results.count)
        return results.prefix(topK).map { result in
            SearchResult(id: "", score: result.distance, internalID: result.nodeID)
        }
    }

    private static func computeDistancesOnGPU(
        context: MetalContext,
        queryBuffer: MTLBuffer,
        vectors: any VectorStorage,
        neighborIDs: [UInt32],
        metric: Metric
    ) async throws -> [Float] {
        guard !neighborIDs.isEmpty else {
            return []
        }

        let functionName: String
        switch metric {
        case .cosine:
            functionName = vectors.isFloat16 ? "cosine_distance_indexed_f16" : "cosine_distance_indexed"
        case .l2:
            functionName = vectors.isFloat16 ? "l2_distance_indexed_f16" : "l2_distance_indexed"
        case .innerProduct:
            functionName = vectors.isFloat16 ? "inner_product_distance_indexed_f16" : "inner_product_distance_indexed"
        case .hamming:
            throw ANNSError.searchFailed("SearchGPU does not support metric .hamming")
        }

        let pipeline = try await context.pipelineCache.pipeline(for: functionName)

        let neighborLength = neighborIDs.count * MemoryLayout<UInt32>.stride
        let outputLength = neighborIDs.count * MemoryLayout<Float>.stride
        let workspace = try acquireWorkspace(
            device: context.device,
            neighborBytes: neighborLength,
            outputBytes: outputLength
        )
        defer { workspacePool.release(workspace) }
        copy(neighborIDs, into: workspace.neighborBuffer, byteCount: neighborLength)

        var dim = UInt32(vectors.dim)
        var count = UInt32(neighborIDs.count)

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.searchFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 1)
            encoder.setBuffer(workspace.neighborBuffer, offset: 0, index: 2)
            encoder.setBuffer(workspace.outputBuffer, offset: 0, index: 3)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadsPerGrid = MTLSize(width: neighborIDs.count, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(neighborIDs.count, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        let outputPointer = workspace.outputBuffer.contents().bindMemory(to: Float.self, capacity: neighborIDs.count)
        return Array(UnsafeBufferPointer(start: outputPointer, count: neighborIDs.count))
    }

    private static func makeQueryBuffer(
        context: MetalContext,
        query: [Float],
        vectors: any VectorStorage
    ) throws -> MTLBuffer {
        if vectors.isFloat16 {
            var queryHalf: [UInt16] = []
            queryHalf.reserveCapacity(query.count)
            for value in query {
                queryHalf.append(Float16(value).bitPattern)
            }

            let queryLength = queryHalf.count * MemoryLayout<UInt16>.stride
            guard
                let buffer = context.device.makeBuffer(
                    bytes: queryHalf,
                    length: queryLength,
                    options: .storageModeShared
                )
            else {
                throw ANNSError.searchFailed("Failed to allocate Float16 query buffer")
            }
            return buffer
        }

        let queryLength = query.count * MemoryLayout<Float>.stride
        guard
            let buffer = context.device.makeBuffer(
                bytes: query,
                length: queryLength,
                options: .storageModeShared
            )
        else {
            throw ANNSError.searchFailed("Failed to allocate Float32 query buffer")
        }
        return buffer
    }

    private static func insertSorted(_ candidate: Candidate, into list: inout [Candidate]) {
        let insertionIndex = lowerBound(of: candidate.distance, in: list)
        list.insert(candidate, at: insertionIndex)
    }

    private static func lowerBound(of distance: Float, in list: [Candidate]) -> Int {
        var low = 0
        var high = list.count

        while low < high {
            let mid = (low + high) / 2
            if list[mid].distance < distance {
                low = mid + 1
            } else {
                high = mid
            }
        }

        return low
    }

    private static func acquireWorkspace(
        device: MTLDevice,
        neighborBytes: Int,
        outputBytes: Int
    ) throws -> Workspace {
        let deviceID = ObjectIdentifier(device)
        return try workspacePool.acquire(
            where: {
                $0.deviceID == deviceID && $0.neighborBuffer.length >= max(neighborBytes, 1)
                    && $0.outputBuffer.length >= max(outputBytes, 1)
            },
            make: {
                try Workspace.allocate(device: device, neighborBytes: neighborBytes, outputBytes: outputBytes)
            }
        )
    }

    private static func copy<T>(_ source: [T], into buffer: MTLBuffer, byteCount: Int) {
        source.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress, byteCount > 0 else {
                return
            }
            buffer.contents().copyMemory(from: baseAddress, byteCount: byteCount)
        }
    }

    // Immutable after init; holds only MTLBuffer references.
    private final class Workspace: @unchecked Sendable {
        let deviceID: ObjectIdentifier
        let neighborBuffer: MTLBuffer
        let outputBuffer: MTLBuffer

        init(deviceID: ObjectIdentifier, neighborBuffer: MTLBuffer, outputBuffer: MTLBuffer) {
            self.deviceID = deviceID
            self.neighborBuffer = neighborBuffer
            self.outputBuffer = outputBuffer
        }

        static func allocate(
            device: MTLDevice,
            neighborBytes: Int,
            outputBytes: Int
        ) throws -> Workspace {
            guard
                let neighborBuffer = device.makeBuffer(length: max(neighborBytes, 1), options: .storageModeShared),
                let outputBuffer = device.makeBuffer(length: max(outputBytes, 1), options: .storageModeShared)
            else {
                throw ANNSError.searchFailed("Failed to allocate SearchGPU workspace buffers")
            }

            return Workspace(
                deviceID: ObjectIdentifier(device),
                neighborBuffer: neighborBuffer,
                outputBuffer: outputBuffer
            )
        }
    }
}
