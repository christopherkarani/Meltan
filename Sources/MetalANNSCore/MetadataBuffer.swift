import Foundation
import Metal

/// GPU-accessible metadata stored as 5 UInt32 fields in one Metal buffer.
/// Layout: [entryPointID, nodeCount, degree, dim, iterationCount]
// Thread-safety: All mutable operations and reads are synchronized via an internal NSLock.
//
//
//
package final class MetadataBuffer: @unchecked Sendable {
    package let buffer: MTLBuffer
    private let pointer: UnsafeMutablePointer<UInt32>

    package init(device: MTLDevice? = nil) throws {
        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let byteLength = 5 * MemoryLayout<UInt32>.stride
        guard let buffer = metalDevice.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate MetadataBuffer")
        }

        self.buffer = buffer
        self.pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: 5)
        memset(buffer.contents(), 0, byteLength)
    }

    package var entryPointID: UInt32 {
        get { pointer[0] }
        set { pointer[0] = newValue }
    }

    package var nodeCount: UInt32 {
        get { pointer[1] }
        set { pointer[1] = newValue }
    }

    package var degree: UInt32 {
        get { pointer[2] }
        set { pointer[2] = newValue }
    }

    package var dim: UInt32 {
        get { pointer[3] }
        set { pointer[3] = newValue }
    }

    package var iterationCount: UInt32 {
        get { pointer[4] }
        set { pointer[4] = newValue }
    }
}
