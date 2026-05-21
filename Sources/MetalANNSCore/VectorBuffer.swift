import Foundation
import Metal

/// GPU-resident flat buffer storing `capacity` vectors of `dim` dimensions.
/// Layout: vector[i] starts at offset `i * dim` in the underlying Float32 buffer.
/// GPU-resident flat buffer storing `capacity` vectors of `dim` dimensions.
/// Layout: vector[i] starts at offset `i * dim` in the underlying Float32 buffer.
///
/// Thread-safety: All mutable operations (`insert`, `setCount`) and reads (`vector(at:)`)
/// are synchronized via an internal `NSLock`. This makes `VectorBuffer` safe for concurrent
/// access without relying solely on the owning actor's serialization.
public final class VectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let dim: Int
    public let capacity: Int
    public var count: Int { lock.withLock { _count } }

    private let rawPointer: UnsafeMutablePointer<Float>
    private let lock = NSLock()
    private var _count: Int = 0

    public init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws {
        guard capacity >= 0, dim > 0 else {
            throw ANNSError.constructionFailed("VectorBuffer requires capacity >= 0 and dim > 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let elementCount = capacity * dim
        let byteLength = elementCount * MemoryLayout<Float>.stride

        guard let buffer = metalDevice.makeBuffer(length: max(byteLength, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate VectorBuffer")
        }

        self.buffer = buffer
        self.dim = dim
        self.capacity = capacity
        self.rawPointer = buffer.contents().bindMemory(to: Float.self, capacity: max(elementCount, 1))
    }

    public func setCount(_ newCount: Int) {
        lock.withLock { _count = newCount }
    }

    public func insert(vector: [Float], at index: Int) throws {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) is out of bounds for capacity \(capacity)")
        }

        let offset = index * dim
        lock.withLock {
            vector.withUnsafeBufferPointer { source in
                guard let baseAddress = source.baseAddress else { return }
                rawPointer.advanced(by: offset).update(from: baseAddress, count: dim)
            }
        }
    }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        lock.withLock {
            for (offset, vector) in vectors.enumerated() {
                let index = start + offset
                let vecOffset = index * dim
                vector.withUnsafeBufferPointer { source in
                    guard let baseAddress = source.baseAddress else { return }
                    rawPointer.advanced(by: vecOffset).update(from: baseAddress, count: dim)
                }
            }
        }
    }

    public func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < capacity, "Index out of bounds")
        let offset = index * dim
        return lock.withLock {
            let pointer = rawPointer.advanced(by: offset)
            return Array(UnsafeBufferPointer(start: pointer, count: dim))
        }
    }

    public var floatPointer: UnsafeBufferPointer<Float> {
        lock.withLock {
            UnsafeBufferPointer(start: rawPointer, count: capacity * dim)
        }
    }
}

extension VectorBuffer: VectorStorage {
    public var isFloat16: Bool { false }
}
