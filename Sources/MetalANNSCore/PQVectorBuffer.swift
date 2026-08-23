import Foundation

package final class PQVectorBuffer: QuantizedStorage, @unchecked Sendable {
    package let originalDimension: Int
    package private(set) var count: Int
    package let capacity: Int

    private let pq: ProductQuantizer
    private let codeLength: Int
    private var codes: [UInt8]

    package init(capacity: Int, dim: Int, pq: ProductQuantizer) throws {
        guard capacity > 0 else {
            throw ANNSError.constructionFailed("PQVectorBuffer capacity must be greater than zero")
        }
        let expectedDimension = pq.numSubspaces * pq.subspaceDimension
        guard dim == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: dim)
        }

        self.originalDimension = dim
        self.count = 0
        self.capacity = capacity
        self.pq = pq
        self.codeLength = pq.numSubspaces
        self.codes = [UInt8](repeating: 0, count: capacity * pq.numSubspaces)
    }

    package func insert(vector: [Float], at index: Int) throws {
        let encoded = try pq.encode(vector: vector)
        try insertEncoded(code: encoded, at: index)
    }

    package func insertEncoded(code: [UInt8], at index: Int) throws {
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) out of bounds for capacity \(capacity)")
        }
        guard code.count == codeLength else {
            throw ANNSError.dimensionMismatch(expected: codeLength, got: code.count)
        }

        let offset = index * codeLength
        codes.withUnsafeMutableBufferPointer { destination in
            code.withUnsafeBufferPointer { source in
                destination.baseAddress!.advanced(by: offset).update(from: source.baseAddress!, count: codeLength)
            }
        }
        if count < index + 1 {
            count = index + 1
        }
    }

    package func approximateDistance(query: [Float], to index: UInt32, metric: Metric) -> Float {
        guard query.count == originalDimension else {
            return Float.greatestFiniteMagnitude
        }
        let indexInt = Int(index)
        guard indexInt >= 0, indexInt < count else {
            return Float.greatestFiniteMagnitude
        }
        return pq.approximateDistance(query: query, codes: code(at: indexInt), metric: metric)
    }

    package func reconstruct(at index: UInt32) -> [Float] {
        let indexInt = Int(index)
        guard indexInt >= 0, indexInt < count else {
            return []
        }
        do {
            return try pq.reconstruct(codes: code(at: indexInt))
        } catch {
            return []
        }
    }

    package func code(at index: Int) -> [UInt8] {
        guard index >= 0, index < count else {
            return []
        }
        let offset = index * codeLength
        return Array(codes[offset..<(offset + codeLength)])
    }

    package func withCode<R>(at index: Int, _ body: (UnsafeBufferPointer<UInt8>) -> R) -> R? {
        guard index >= 0, index < count else {
            return nil
        }
        let offset = index * codeLength
        return codes.withUnsafeBufferPointer { buffer in
            let slice = UnsafeBufferPointer(
                start: buffer.baseAddress!.advanced(by: offset),
                count: codeLength
            )
            return body(slice)
        }
    }

    package func gatherCodes(for ids: [UInt32]) -> [UInt8] {
        guard !ids.isEmpty else {
            return []
        }

        var flattened = [UInt8](repeating: 0, count: ids.count * codeLength)
        flattened.withUnsafeMutableBufferPointer { destination in
            codes.withUnsafeBufferPointer { source in
                for (row, id) in ids.enumerated() {
                    let index = Int(id)
                    guard index >= 0, index < count else {
                        continue
                    }

                    let sourceOffset = index * codeLength
                    let destinationOffset = row * codeLength
                    destination.baseAddress!.advanced(by: destinationOffset).update(
                        from: source.baseAddress!.advanced(by: sourceOffset),
                        count: codeLength
                    )
                }
            }
        }
        return flattened
    }

    package var compressedCodeBytes: Int {
        count * codeLength
    }
}
