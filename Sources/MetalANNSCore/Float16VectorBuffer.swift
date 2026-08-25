import Foundation
import Metal

/// GPU-resident flat buffer storing `capacity` vectors of `dim` dimensions in Float16.
/// Layout: vector[i] starts at offset `i * dim` in the underlying half buffer.
/// Converts `[Float]` <-> Float16 at the API boundary.
package final class Float16VectorBuffer: @unchecked Sendable {
    package let buffer: MTLBuffer
    package let dim: Int
    package let capacity: Int
    package private(set) var count: Int = 0

    private let rawPointer: UnsafeMutablePointer<UInt16>

    package init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws {
        guard capacity >= 0, dim > 0 else {
            throw ANNSError.constructionFailed("Float16VectorBuffer requires capacity >= 0 and dim > 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let elementCount = capacity * dim
        let byteLength = elementCount * MemoryLayout<UInt16>.stride

        guard let buffer = metalDevice.makeBuffer(length: max(byteLength, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate Float16VectorBuffer")
        }

        self.buffer = buffer
        self.dim = dim
        self.capacity = capacity
        self.rawPointer = buffer.contents().bindMemory(to: UInt16.self, capacity: max(elementCount, 1))
    }

    package func setCount(_ newCount: Int) {
        count = newCount
    }

    package func insert(vector: [Float], at index: Int) throws {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) is out of bounds for capacity \(capacity)")
        }

        let offset = index * dim
        for d in 0..<dim {
            rawPointer[offset + d] = Float16(vector[d]).bitPattern
        }
    }

    package func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        for (offset, vector) in vectors.enumerated() {
            try insert(vector: vector, at: start + offset)
        }
    }

    package func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < capacity, "Index out of bounds")
        let offset = index * dim
        var result = [Float](repeating: 0, count: dim)
        for d in 0..<dim {
            result[d] = Float(Float16(bitPattern: rawPointer[offset + d]))
        }
        return result
    }

    package var halfPointer: UnsafeBufferPointer<UInt16> {
        UnsafeBufferPointer(start: rawPointer, count: capacity * dim)
    }
}

extension Float16VectorBuffer: VectorStorage {
    package var isFloat16: Bool { true }
}
