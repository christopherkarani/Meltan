import Foundation
import Metal

/// Thread-safe pool of reusable MTLBuffer triplets for GPU search operations.
/// Eliminates per-search allocation overhead in FullGPUSearch.
// Thread-safety: buffer storage is delegated to the NSLock-guarded
// WorkspaceBufferPool; the visited-generation counter is guarded by generationLock.
package final class SearchBufferPool: @unchecked Sendable {
    package struct Buffers: @unchecked Sendable {
        package let queryBuffer: MTLBuffer
        package let outputDistanceBuffer: MTLBuffer
        package let outputIDBuffer: MTLBuffer
        package let queryDim: Int
        package let maxK: Int
    }

    package struct VisitedBuffers: @unchecked Sendable {
        package let buffer: MTLBuffer
        package let generation: UInt32
    }

    private let device: MTLDevice
    private let bufferPool: WorkspaceBufferPool<Buffers>
    private let visitedPool: WorkspaceBufferPool<(buffer: MTLBuffer, capacity: Int)>
    private let generationLock = NSLock()
    private var generationCounter: UInt32 = 0

    package init(
        device: MTLDevice,
        maxRetainedEntries: Int = 32,
        maxRetainedBytes: Int = 64 * 1024 * 1024
    ) {
        self.device = device
        self.bufferPool = WorkspaceBufferPool(
            maxRetainedEntries: maxRetainedEntries,
            maxRetainedBytes: maxRetainedBytes,
            entryBytes: Self.entryBytes
        )
        self.visitedPool = WorkspaceBufferPool(
            maxRetainedEntries: .max,
            entryBytes: { $0.buffer.length }
        )
    }

    /// Returns a buffer set with capacity >= requested dimensions.
    /// If no pooled entry fits, allocates new buffers.
    package func acquire(queryDim: Int, maxK: Int) throws -> Buffers {
        try bufferPool.acquire(
            where: { $0.queryDim >= queryDim && $0.maxK >= maxK },
            make: { try Self.allocate(device: device, queryDim: queryDim, maxK: maxK) }
        )
    }

    /// Returns buffers to the pool for future reuse.
    package func release(_ buffers: Buffers) {
        bufferPool.release(buffers)
    }

    /// Acquires a visited-generation buffer sized for `nodeCount` nodes.
    /// Returns a pooled or newly allocated buffer and a unique non-zero generation value.
    /// Reused buffers are intentionally not zeroed; generations provide per-search isolation.
    package func acquireVisited(nodeCount: Int) throws -> VisitedBuffers {
        let generation = nextGeneration()
        let capacity = max(nodeCount, 1)

        let entry = try visitedPool.acquire(
            where: { $0.capacity >= capacity },
            make: { try Self.allocateVisited(device: device, capacity: capacity) }
        )
        return VisitedBuffers(buffer: entry.buffer, generation: generation)
    }

    /// Returns a visited-generation buffer to the pool.
    /// The provided capacity should match the node count used during acquire.
    package func releaseVisited(_ buffer: MTLBuffer, capacity: Int) {
        visitedPool.release((buffer: buffer, capacity: max(capacity, 1)))
    }

    var availableCountForTesting: Int {
        bufferPool.availableCountForTesting
    }

    var retainedBytesForTesting: Int {
        bufferPool.retainedBytesForTesting
    }

    private func nextGeneration() -> UInt32 {
        generationLock.lock()
        defer { generationLock.unlock() }
        generationCounter = generationCounter == UInt32.max ? 1 : generationCounter + 1
        return generationCounter
    }

    private static func allocate(device: MTLDevice, queryDim: Int, maxK: Int) throws -> Buffers {
        let floatSize = MemoryLayout<Float>.stride
        let uintSize = MemoryLayout<UInt32>.stride

        guard
            let qBuf = device.makeBuffer(length: queryDim * floatSize, options: .storageModeShared),
            let dBuf = device.makeBuffer(length: max(maxK * floatSize, floatSize), options: .storageModeShared),
            let iBuf = device.makeBuffer(length: max(maxK * uintSize, uintSize), options: .storageModeShared)
        else {
            throw ANNSError.searchFailed("Failed to allocate search buffer pool entry")
        }

        return Buffers(
            queryBuffer: qBuf,
            outputDistanceBuffer: dBuf,
            outputIDBuffer: iBuf,
            queryDim: queryDim,
            maxK: maxK
        )
    }

    private static func allocateVisited(
        device: MTLDevice,
        capacity: Int
    ) throws -> (buffer: MTLBuffer, capacity: Int) {
        let length = max(capacity * MemoryLayout<UInt32>.stride, MemoryLayout<UInt32>.stride)
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw ANNSError.searchFailed("Failed to allocate visited generation buffer")
        }

        buffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: capacity)
        return (buffer: buffer, capacity: capacity)
    }

    private static func entryBytes(_ buffers: Buffers) -> Int {
        buffers.queryBuffer.length + buffers.outputDistanceBuffer.length + buffers.outputIDBuffer.length
    }
}
