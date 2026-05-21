import Foundation
import Metal

/// GPU-resident adjacency list stored as two flat 2D arrays:
/// `adjacency[nodeID * degree + slot]` and `distances[nodeID * degree + slot]`.
///
/// Thread-safety: All mutable operations (`setNeighbors`, `setCount`) and reads
/// (`neighborIDs`, `neighborDistances`) are synchronized via an internal `NSLock`.
public final class GraphBuffer: @unchecked Sendable {
    public let adjacencyBuffer: MTLBuffer
    public let distanceBuffer: MTLBuffer
    public let degree: Int
    public let capacity: Int
    public var nodeCount: Int { lock.withLock { _nodeCount } }

    private let idPointer: UnsafeMutablePointer<UInt32>
    private let distPointer: UnsafeMutablePointer<Float>
    private let lock = NSLock()
    private var _nodeCount: Int = 0

    public init(capacity: Int, degree: Int, device: MTLDevice? = nil) throws {
        guard capacity >= 0, degree > 0 else {
            throw ANNSError.constructionFailed("GraphBuffer requires capacity >= 0 and degree > 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let slotCount = capacity * degree
        let adjacencyBytes = slotCount * MemoryLayout<UInt32>.stride
        let distanceBytes = slotCount * MemoryLayout<Float>.stride

        guard let adjacencyBuffer = metalDevice.makeBuffer(length: max(adjacencyBytes, 4), options: .storageModeShared),
              let distanceBuffer = metalDevice.makeBuffer(length: max(distanceBytes, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate GraphBuffer")
        }

        self.adjacencyBuffer = adjacencyBuffer
        self.distanceBuffer = distanceBuffer
        self.degree = degree
        self.capacity = capacity
        self.idPointer = adjacencyBuffer.contents().bindMemory(to: UInt32.self, capacity: max(slotCount, 1))
        self.distPointer = distanceBuffer.contents().bindMemory(to: Float.self, capacity: max(slotCount, 1))

        for index in 0..<slotCount {
            idPointer[index] = UInt32.max
            distPointer[index] = Float.greatestFiniteMagnitude
        }
    }

    /// Creates a graph view over pre-existing Metal buffers (e.g. mmap-backed bytesNoCopy buffers).
    /// Buffers must contain `capacity * degree` entries each.
    public init(
        adjacencyBuffer: MTLBuffer,
        distanceBuffer: MTLBuffer,
        capacity: Int,
        degree: Int,
        nodeCount: Int
    ) throws {
        guard capacity >= 0, degree > 0 else {
            throw ANNSError.constructionFailed("GraphBuffer requires capacity >= 0 and degree > 0")
        }
        guard nodeCount >= 0, nodeCount <= capacity else {
            throw ANNSError.constructionFailed("Node count is out of bounds for capacity")
        }

        let slotCount = capacity * degree
        let adjacencyBytes = slotCount * MemoryLayout<UInt32>.stride
        let distanceBytes = slotCount * MemoryLayout<Float>.stride

        guard adjacencyBuffer.length >= max(adjacencyBytes, 4),
              distanceBuffer.length >= max(distanceBytes, 4) else {
            throw ANNSError.constructionFailed("Provided buffers are too small for GraphBuffer layout")
        }

        self.adjacencyBuffer = adjacencyBuffer
        self.distanceBuffer = distanceBuffer
        self.degree = degree
        self.capacity = capacity
        self._nodeCount = nodeCount
        self.idPointer = adjacencyBuffer.contents().bindMemory(to: UInt32.self, capacity: max(slotCount, 1))
        self.distPointer = distanceBuffer.contents().bindMemory(to: Float.self, capacity: max(slotCount, 1))
    }

    public func setCount(_ newCount: Int) {
        lock.withLock { _nodeCount = newCount }
    }

    public func setNeighbors(of nodeID: Int, ids: [UInt32], distances: [Float]) throws {
        guard ids.count == degree, distances.count == degree else {
            throw ANNSError.constructionFailed("Neighbor count must equal degree \(degree)")
        }
        guard nodeID >= 0, nodeID < capacity else {
            throw ANNSError.constructionFailed("Node ID \(nodeID) is out of bounds for capacity \(capacity)")
        }

        let base = nodeID * degree
        lock.withLock {
            for slot in 0..<degree {
                idPointer[base + slot] = ids[slot]
                distPointer[base + slot] = distances[slot]
            }
        }
    }

    public func neighborIDs(of nodeID: Int) -> [UInt32] {
        precondition(nodeID >= 0 && nodeID < capacity, "Node ID out of bounds")
        let base = nodeID * degree
        return lock.withLock {
            Array(UnsafeBufferPointer(start: idPointer.advanced(by: base), count: degree))
        }
    }

    public func neighborDistances(of nodeID: Int) -> [Float] {
        precondition(nodeID >= 0 && nodeID < capacity, "Node ID out of bounds")
        let base = nodeID * degree
        return lock.withLock {
            Array(UnsafeBufferPointer(start: distPointer.advanced(by: base), count: degree))
        }
    }
}
