import Foundation
import Metal

package enum GPUADCSearch {
    private static let scanLoadStride = 32
    private static let workspacePool = WorkspaceBufferPool<Workspace>(
        maxRetainedEntries: 8,
        entryBytes: {
            $0.queryBuffer.length + $0.codebookBuffer.length + $0.distanceTableBuffer.length
                + $0.codesBuffer.length + $0.distancesBuffer.length
        }
    )

    package static func computeDistances(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        codes: [[UInt8]],
        flatCodebooks: [Float]? = nil
    ) async throws -> [Float] {
        let m = pq.numSubspaces
        var packedCodes: [UInt8] = []
        packedCodes.reserveCapacity(codes.count * m)
        for (index, code) in codes.enumerated() {
            guard code.count == m else {
                throw ANNSError.searchFailed("Invalid PQ code size at index \(index)")
            }
            packedCodes.append(contentsOf: code)
        }

        return try await computeDistances(
            context: context,
            query: query,
            pq: pq,
            packedCodes: packedCodes,
            vectorCount: codes.count,
            flatCodebooks: flatCodebooks
        )
    }

    package static func computeDistances(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        packedCodes: [UInt8],
        vectorCount: Int,
        flatCodebooks: [Float]? = nil
    ) async throws -> [Float] {
        let expectedDimension = pq.numSubspaces * pq.subspaceDimension
        guard query.count == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: query.count)
        }
        guard vectorCount >= 0 else {
            throw ANNSError.searchFailed("Vector count must be non-negative")
        }
        guard vectorCount > 0 else {
            return []
        }

        let m = pq.numSubspaces
        let ks = pq.centroidsPerSubspace
        let subspaceDim = pq.subspaceDimension
        guard ks <= Int(UInt8.max) + 1 else {
            throw ANNSError.searchFailed("GPU ADC supports at most 256 centroids per subspace")
        }
        guard packedCodes.count == vectorCount * m else {
            throw ANNSError.searchFailed(
                "Packed PQ code length mismatch. Expected \(vectorCount * m), got \(packedCodes.count)"
            )
        }
        for (index, value) in packedCodes.enumerated() {
            guard Int(value) < ks else {
                throw ANNSError.searchFailed("PQ code value out of range at packed index \(index)")
            }
        }

        let originalVectorCount = vectorCount
        let paddedVectorCount = roundUp(originalVectorCount, toMultipleOf: scanLoadStride)

        let flattenedCodebooks = flatCodebooks ?? flattenCodebooks(from: pq)
        let expectedCodebookCount = m * ks * subspaceDim
        guard flattenedCodebooks.count == expectedCodebookCount else {
            throw ANNSError.searchFailed(
                "Invalid flattened codebook size. Expected \(expectedCodebookCount), got \(flattenedCodebooks.count)"
            )
        }

        let tableLengthBytes = m * ks * MemoryLayout<Float>.stride
        guard tableLengthBytes <= context.device.maxThreadgroupMemoryLength else {
            throw ANNSError.searchFailed(
                "PQ distance table (\(tableLengthBytes) bytes) exceeds device threadgroup memory limit "
                    + "(\(context.device.maxThreadgroupMemoryLength) bytes). "
                    + "Reduce M (current: \(m)) or Ks (current: \(ks))."
            )
        }
        let distancesLengthBytes = paddedVectorCount * MemoryLayout<Float>.stride
        let queryLengthBytes = query.count * MemoryLayout<Float>.stride
        let codebookLengthBytes = flattenedCodebooks.count * MemoryLayout<Float>.stride
        let codesLengthBytes = paddedVectorCount * m * MemoryLayout<UInt8>.stride

        let workspace = try acquireWorkspace(
            device: context.device,
            queryBytes: queryLengthBytes,
            codebookBytes: codebookLengthBytes,
            tableBytes: tableLengthBytes,
            codesBytes: codesLengthBytes,
            distancesBytes: distancesLengthBytes
        )
        defer { workspacePool.release(workspace) }

        copy(query, into: workspace.queryBuffer, byteCount: queryLengthBytes)
        copy(flattenedCodebooks, into: workspace.codebookBuffer, byteCount: codebookLengthBytes)
        copy(packedCodes, into: workspace.codesBuffer, byteCount: packedCodes.count)
        if paddedVectorCount > originalVectorCount {
            let paddingOffset = packedCodes.count
            let paddingBytes = (paddedVectorCount - originalVectorCount) * m
            workspace.codesBuffer.contents().advanced(by: paddingOffset).initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: paddingBytes
            )
        }

        let tablePipeline = try await context.pipelineCache.pipeline(for: "pq_compute_distance_table")
        let scanPipeline = try await context.pipelineCache.pipeline(for: "pq_adc_scan")

        var mU32 = UInt32(m)
        var ksU32 = UInt32(ks)
        var subspaceDimU32 = UInt32(subspaceDim)
        var vectorCountU32 = UInt32(paddedVectorCount)

        guard scanPipeline.maxTotalThreadsPerThreadgroup >= scanLoadStride else {
            throw ANNSError.searchFailed(
                "pq_adc_scan requires at least \(scanLoadStride) threads per threadgroup"
            )
        }

        try await context.execute { commandBuffer in
            guard let tableEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.gpuPipelineUnavailable("Failed to create distance-table encoder")
            }

            tableEncoder.setComputePipelineState(tablePipeline)
            tableEncoder.setBuffer(workspace.queryBuffer, offset: 0, index: 0)
            tableEncoder.setBuffer(workspace.codebookBuffer, offset: 0, index: 1)
            tableEncoder.setBuffer(workspace.distanceTableBuffer, offset: 0, index: 2)
            tableEncoder.setBytes(&mU32, length: MemoryLayout<UInt32>.stride, index: 3)
            tableEncoder.setBytes(&ksU32, length: MemoryLayout<UInt32>.stride, index: 4)
            tableEncoder.setBytes(&subspaceDimU32, length: MemoryLayout<UInt32>.stride, index: 5)

            let tableGrid = MTLSize(width: m, height: ks, depth: 1)
            let tableThreads = MTLSize(width: 8, height: 8, depth: 1)
            tableEncoder.dispatchThreads(tableGrid, threadsPerThreadgroup: tableThreads)
            tableEncoder.endEncoding()

            guard let scanEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.gpuPipelineUnavailable("Failed to create ADC scan encoder")
            }

            scanEncoder.setComputePipelineState(scanPipeline)
            scanEncoder.setBuffer(workspace.codesBuffer, offset: 0, index: 0)
            scanEncoder.setBuffer(workspace.distanceTableBuffer, offset: 0, index: 1)
            scanEncoder.setBuffer(workspace.distancesBuffer, offset: 0, index: 2)
            scanEncoder.setBytes(&mU32, length: MemoryLayout<UInt32>.stride, index: 3)
            scanEncoder.setBytes(&ksU32, length: MemoryLayout<UInt32>.stride, index: 4)
            scanEncoder.setBytes(&vectorCountU32, length: MemoryLayout<UInt32>.stride, index: 5)
            scanEncoder.setThreadgroupMemoryLength(tableLengthBytes, index: 0)

            let scanGrid = MTLSize(width: paddedVectorCount, height: 1, depth: 1)
            let scanThreadWidth = scanLoadStride
            let scanThreads = MTLSize(width: scanThreadWidth, height: 1, depth: 1)
            scanEncoder.dispatchThreads(scanGrid, threadsPerThreadgroup: scanThreads)
            scanEncoder.endEncoding()
        }

        let base = workspace.distancesBuffer.contents().bindMemory(to: Float.self, capacity: paddedVectorCount)
        let allDistances = Array(UnsafeBufferPointer(start: base, count: paddedVectorCount))
        return Array(allDistances.prefix(originalVectorCount))
    }

    package static func search(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        codes: [[UInt8]],
        ids: [String],
        k: Int,
        flatCodebooks: [Float]? = nil
    ) async throws -> [SearchResult] {
        let distances = try await computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            flatCodebooks: flatCodebooks
        )
        return try rankDistances(distances: distances, ids: ids, k: k)
    }

    package static func flattenCodebooks(from pq: ProductQuantizer) -> [Float] {
        var flattened: [Float] = []
        flattened.reserveCapacity(
            pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension
        )
        for subspace in 0..<pq.numSubspaces {
            for centroid in 0..<pq.centroidsPerSubspace {
                flattened.append(contentsOf: pq.codebooks[subspace][centroid])
            }
        }
        return flattened
    }

    static func rankDistances(
        distances: [Float],
        ids: [String],
        k: Int
    ) throws -> [SearchResult] {
        guard distances.count == ids.count else {
            throw ANNSError.constructionFailed("codes and ids count mismatch")
        }
        guard k > 0 else {
            return []
        }
        guard !distances.isEmpty else {
            return []
        }

        var topResults = BoundedPriorityBuffer<SearchResult>(capacity: min(k, distances.count)) {
            lhs,
            rhs in
            lhs.score < rhs.score
        }
        for (index, distance) in distances.enumerated() {
            topResults.insert(SearchResult(id: ids[index], score: distance, internalID: UInt32(index)))
        }
        return topResults.sortedElements()
    }

    private static func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private static func copy<T>(_ source: [T], into buffer: MTLBuffer, byteCount: Int) {
        source.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress, byteCount > 0 else {
                return
            }
            buffer.contents().copyMemory(from: baseAddress, byteCount: byteCount)
        }
    }

    private static func acquireWorkspace(
        device: MTLDevice,
        queryBytes: Int,
        codebookBytes: Int,
        tableBytes: Int,
        codesBytes: Int,
        distancesBytes: Int
    ) throws -> Workspace {
        let deviceID = ObjectIdentifier(device)
        return try workspacePool.acquire(
            where: { workspace in
                workspace.deviceID == deviceID && workspace.queryBuffer.length >= queryBytes
                    && workspace.codebookBuffer.length >= codebookBytes
                    && workspace.distanceTableBuffer.length >= tableBytes && workspace.codesBuffer.length >= codesBytes
                    && workspace.distancesBuffer.length >= distancesBytes
            },
            make: {
                try Workspace.allocate(
                    device: device,
                    queryBytes: queryBytes,
                    codebookBytes: codebookBytes,
                    tableBytes: tableBytes,
                    codesBytes: codesBytes,
                    distancesBytes: distancesBytes
                )
            }
        )
    }

    private final class Workspace: @unchecked Sendable {
        let deviceID: ObjectIdentifier
        let queryBuffer: MTLBuffer
        let codebookBuffer: MTLBuffer
        let distanceTableBuffer: MTLBuffer
        let codesBuffer: MTLBuffer
        let distancesBuffer: MTLBuffer

        init(
            deviceID: ObjectIdentifier,
            queryBuffer: MTLBuffer,
            codebookBuffer: MTLBuffer,
            distanceTableBuffer: MTLBuffer,
            codesBuffer: MTLBuffer,
            distancesBuffer: MTLBuffer
        ) {
            self.deviceID = deviceID
            self.queryBuffer = queryBuffer
            self.codebookBuffer = codebookBuffer
            self.distanceTableBuffer = distanceTableBuffer
            self.codesBuffer = codesBuffer
            self.distancesBuffer = distancesBuffer
        }

        static func allocate(
            device: MTLDevice,
            queryBytes: Int,
            codebookBytes: Int,
            tableBytes: Int,
            codesBytes: Int,
            distancesBytes: Int
        ) throws -> Workspace {
            guard
                let queryBuffer = device.makeBuffer(length: max(queryBytes, 1), options: .storageModeShared),
                let codebookBuffer = device.makeBuffer(length: max(codebookBytes, 1), options: .storageModeShared),
                let distanceTableBuffer = device.makeBuffer(length: max(tableBytes, 1), options: .storageModeShared),
                let codesBuffer = device.makeBuffer(length: max(codesBytes, 1), options: .storageModeShared),
                let distancesBuffer = device.makeBuffer(length: max(distancesBytes, 1), options: .storageModeShared)
            else {
                throw ANNSError.gpuResourceExhausted("Failed to allocate Metal buffers for GPU ADC")
            }

            return Workspace(
                deviceID: ObjectIdentifier(device),
                queryBuffer: queryBuffer,
                codebookBuffer: codebookBuffer,
                distanceTableBuffer: distanceTableBuffer,
                codesBuffer: codesBuffer,
                distancesBuffer: distancesBuffer
            )
        }
    }
}
