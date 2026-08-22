import Foundation
import Metal

/// Thread-safe pool of reusable MTLBuffer triplets for GPU search operations.
/// Eliminates per-search allocation overhead in FullGPUSearch.
// Thread-safety: All mutable operations and reads are synchronized via an internal NSLock.
//
//
//
public final class SearchBufferPool: @unchecked Sendable {
    public struct Buffers: @unchecked Sendable {
        public let queryBuffer: MTLBuffer
        public let outputDistanceBuffer: MTLBuffer
        public let outputIDBuffer: MTLBuffer
        public let queryDim: Int
        public let maxK: Int
    }

    public struct VisitedBuffers: @unchecked Sendable {
        public let buffer: MTLBuffer
        public let generation: UInt32
    }

    private let device: MTLDevice
    private let maxRetainedEntries: Int
    private let maxRetainedBytes: Int
    private var available: [Buffers] = []
    private var visitedAvailable: [(buffer: MTLBuffer, capacity: Int)] = []
    private var retainedBytes: Int = 0
    private var generationCounter: UInt32 = 0
    private let lock = NSLock()

    public init(
        device: MTLDevice,
        maxRetainedEntries: Int = 32,
        maxRetainedBytes: Int = 64 * 1024 * 1024
    ) {
        self.device = device
        self.maxRetainedEntries = max(0, maxRetainedEntries)
        self.maxRetainedBytes = max(0, maxRetainedBytes)
    }

    /// Returns a buffer set with capacity >= requested dimensions.
    /// If no pooled entry fits, allocates new buffers.
    public func acquire(queryDim: Int, maxK: Int) throws -> Buffers {
        lock.lock()
        defer { lock.unlock() }

        if let index = available.firstIndex(where: {
            $0.queryDim >= queryDim && $0.maxK >= maxK
        }) {
            let buffers = available.remove(at: index)
            retainedBytes -= Self.entryBytes(buffers)
            return buffers
        }

        return try allocate(queryDim: queryDim, maxK: maxK)
    }

    /// Returns buffers to the pool for future reuse.
    public func release(_ buffers: Buffers) {
        lock.lock()
        defer { lock.unlock() }

        let bytes = Self.entryBytes(buffers)
        guard
            maxRetainedEntries > 0,
            maxRetainedBytes > 0,
            bytes <= maxRetainedBytes
        else {
            return
        }

        available.append(buffers)
        retainedBytes += bytes
        trimIfNeeded()
    }

    /// Acquires a visited-generation buffer sized for `nodeCount` nodes.
    /// Returns a pooled or newly allocated buffer and a unique non-zero generation value.
    /// Reused buffers are intentionally not zeroed; generations provide per-search isolation.
    public func acquireVisited(nodeCount: Int) throws -> VisitedBuffers {
        lock.lock()
        defer { lock.unlock() }

        generationCounter = generationCounter == UInt32.max ? 1 : generationCounter + 1
        let generation = generationCounter
        let capacity = max(nodeCount, 1)

        if let index = visitedAvailable.firstIndex(where: { $0.capacity >= capacity }) {
            let entry = visitedAvailable.remove(at: index)
            return VisitedBuffers(buffer: entry.buffer, generation: generation)
        }

        let length = max(capacity * MemoryLayout<UInt32>.stride, MemoryLayout<UInt32>.stride)
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw ANNSError.searchFailed("Failed to allocate visited generation buffer")
        }

        buffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: capacity)
        return VisitedBuffers(buffer: buffer, generation: generation)
    }

    /// Returns a visited-generation buffer to the pool.
    /// The provided capacity should match the node count used during acquire.
    public func releaseVisited(_ buffer: MTLBuffer, capacity: Int) {
        lock.lock()
        defer { lock.unlock() }
        visitedAvailable.append((buffer: buffer, capacity: max(capacity, 1)))
    }

    var availableCountForTesting: Int {
        lock.lock()
        defer { lock.unlock() }
        return available.count
    }

    var retainedBytesForTesting: Int {
        lock.lock()
        defer { lock.unlock() }
        return retainedBytes
    }

    private func trimIfNeeded() {
        while available.count > maxRetainedEntries || retainedBytes > maxRetainedBytes {
            let removed = available.removeFirst()
            retainedBytes -= Self.entryBytes(removed)
        }
    }

    private static func entryBytes(_ buffers: Buffers) -> Int {
        buffers.queryBuffer.length + buffers.outputDistanceBuffer.length + buffers.outputIDBuffer.length
    }

    private func allocate(queryDim: Int, maxK: Int) throws -> Buffers {
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
}
