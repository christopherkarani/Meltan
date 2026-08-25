import Foundation
import Metal

/// 1-bit-per-dimension vector storage.
/// Values are binarized as: value >= 0 -> 1, value < 0 -> 0.
// Thread-safety: All mutable operations and reads are synchronized via an internal NSLock.
package final class BinaryVectorBuffer: @unchecked Sendable {
    package let buffer: MTLBuffer
    package let dim: Int
    package let capacity: Int
    package private(set) var count: Int = 0

    package let bytesPerVector: Int

    private let rawPointer: UnsafeMutablePointer<UInt8>

    package init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws {
        guard dim > 0, dim % 8 == 0 else {
            throw ANNSError.constructionFailed(
                "BinaryVectorBuffer requires dim > 0 and dim % 8 == 0, got dim=\(dim)"
            )
        }
        guard capacity >= 0 else {
            throw ANNSError.constructionFailed("BinaryVectorBuffer requires capacity >= 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let bytesPerVector = dim / 8
        let byteLength = max(capacity * bytesPerVector, 4)

        guard let buffer = metalDevice.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate BinaryVectorBuffer")
        }

        self.buffer = buffer
        self.dim = dim
        self.capacity = capacity
        self.bytesPerVector = bytesPerVector
        self.rawPointer = buffer.contents().bindMemory(
            to: UInt8.self,
            capacity: max(capacity * bytesPerVector, 1)
        )
    }

    package func insert(vector: [Float], at index: Int) throws {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) out of bounds for capacity \(capacity)")
        }

        let base = index * bytesPerVector
        for byteIndex in 0..<bytesPerVector {
            var byte: UInt8 = 0
            for bit in 0..<8 {
                let dimIndex = byteIndex * 8 + bit
                if vector[dimIndex] >= 0 {
                    byte |= (1 << (7 - bit))
                }
            }
            rawPointer[base + byteIndex] = byte
        }
    }

    package func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < capacity, "Index out of bounds")
        let base = index * bytesPerVector
        var unpacked = [Float](repeating: 0, count: dim)

        for byteIndex in 0..<bytesPerVector {
            let byte = rawPointer[base + byteIndex]
            for bit in 0..<8 {
                let dimIndex = byteIndex * 8 + bit
                unpacked[dimIndex] = ((byte >> (7 - bit)) & 1) == 1 ? 1.0 : 0.0
            }
        }

        return unpacked
    }

    package func packedVector(at index: Int) -> [UInt8] {
        precondition(index >= 0 && index < capacity, "Index out of bounds")
        let base = index * bytesPerVector
        let start = rawPointer.advanced(by: base)
        return Array(UnsafeBufferPointer(start: start, count: bytesPerVector))
    }
}

extension BinaryVectorBuffer: VectorStorage {
    package var isFloat16: Bool { false }

    package func setCount(_ newCount: Int) {
        count = newCount
    }

    package func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        for (offset, vector) in vectors.enumerated() {
            try insert(vector: vector, at: start + offset)
        }
    }
}
