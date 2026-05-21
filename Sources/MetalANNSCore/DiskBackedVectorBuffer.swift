import Darwin
import Foundation
import Metal

// Thread-safety: All mutable operations and reads are synchronized via an internal NSLock.
// 
// 
// 
public final class DiskBackedVectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let dim: Int
    public let capacity: Int
    public private(set) var count: Int
    public let isFloat16: Bool
    public let isBinary: Bool

    private let mmapPointer: UnsafeRawPointer
    private let dataOffset: Int
    private let bytesPerVector: Int

    private let cacheCapacity: Int
    private var cache: [Int: [Float]] = [:]
    private var cacheOrder: [Int] = []
    private let cacheLock = NSLock()

    public init(
        mmapPointer: UnsafeRawPointer,
        dataOffset: Int,
        dim: Int,
        count: Int,
        isFloat16: Bool,
        isBinary: Bool,
        device: MTLDevice,
        cacheCapacity: Int = 1024
    ) throws {
        self.mmapPointer = mmapPointer
        self.dataOffset = dataOffset
        self.dim = dim
        self.capacity = count
        self.count = count
        self.isFloat16 = isFloat16
        self.isBinary = isBinary
        self.bytesPerVector = if isBinary {
            dim / 8
        } else {
            dim * (isFloat16 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride)
        }
        self.cacheCapacity = max(1, cacheCapacity)

        guard let stagingBuffer = device.makeBuffer(
            length: max(4, bytesPerVector),
            options: .storageModeShared
        ) else {
            throw ANNSError.constructionFailed("Failed to allocate disk-backed staging buffer")
        }
        self.buffer = stagingBuffer
    }

    private func readVector(at index: Int) -> [Float] {
        let byteOffset = dataOffset + index * bytesPerVector

        if isBinary {
            let pointer = mmapPointer.advanced(by: byteOffset).assumingMemoryBound(to: UInt8.self)
            var unpacked = [Float](repeating: 0, count: dim)
            for byteIndex in 0..<bytesPerVector {
                let byte = pointer[byteIndex]
                for bit in 0..<8 {
                    let dimIndex = byteIndex * 8 + bit
                    unpacked[dimIndex] = ((byte >> (7 - bit)) & 1) == 1 ? 1.0 : 0.0
                }
            }
            return unpacked
        } else if isFloat16 {
            let pointer = mmapPointer.advanced(by: byteOffset).assumingMemoryBound(to: UInt16.self)
            var result = [Float](repeating: 0, count: dim)
            for dimIndex in 0..<dim {
                result[dimIndex] = Float(Float16(bitPattern: pointer[dimIndex]))
            }
            return result
        }

        let pointer = mmapPointer.advanced(by: byteOffset).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: dim))
    }

    private func cachedVector(at index: Int) -> [Float] {
        cacheLock.lock()
        defer { cacheLock.unlock() }

        if let cached = cache[index] {
            if let orderIndex = cacheOrder.firstIndex(of: index) {
                cacheOrder.remove(at: orderIndex)
            }
            cacheOrder.append(index)
            return cached
        }

        let vector = readVector(at: index)
        cache[index] = vector
        cacheOrder.append(index)

        if cacheOrder.count > cacheCapacity {
            let evictedIndex = cacheOrder.removeFirst()
            cache[evictedIndex] = nil
        }

        return vector
    }
}

extension DiskBackedVectorBuffer: VectorStorage {
    public func setCount(_ newCount: Int) {
        _ = newCount
    }

    public func insert(vector: [Float], at index: Int) throws {
        _ = vector
        _ = index
        throw ANNSError.constructionFailed("Disk-backed vectors are read-only")
    }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        _ = vectors
        _ = start
        throw ANNSError.constructionFailed("Disk-backed vectors are read-only")
    }

    public func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < count, "Index out of bounds")
        return cachedVector(at: index)
    }
}

public enum DiskBackedIndexLoader {
    private static let headerMagic: [UInt8] = [0x4D, 0x41, 0x4E, 0x4E] // "MANN"
    private static let mmapVersion: UInt32 = 3
    private static var pageSize: Int { max(1, Int(getpagesize())) }

    public struct LoadResult {
        public let vectors: DiskBackedVectorBuffer
        public let graph: GraphBuffer
        public let idMap: IDMap
        public let entryPoint: UInt32
        public let metric: Metric
        public let isBinary: Bool
        public let mmapLifetime: AnyObject

        public init(
            vectors: DiskBackedVectorBuffer,
            graph: GraphBuffer,
            idMap: IDMap,
            entryPoint: UInt32,
            metric: Metric,
            isBinary: Bool,
            mmapLifetime: AnyObject
        ) {
            self.vectors = vectors
            self.graph = graph
            self.idMap = idMap
            self.entryPoint = entryPoint
            self.metric = metric
            self.isBinary = isBinary
            self.mmapLifetime = mmapLifetime
        }
    }

    public static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> LoadResult {
        let region = try DiskBackedMmapRegion(fileURL: fileURL)

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        var cursor = 0
        let magic = try readBytes(
            from: UnsafeRawPointer(region.pointer),
            length: region.length,
            offset: cursor,
            count: 4
        )
        cursor += 4
        guard magic == headerMagic else {
            throw ANNSError.corruptFile("Invalid file magic")
        }

        let formatVersion = try readUInt32(
            from: UnsafeRawPointer(region.pointer),
            length: region.length,
            cursor: &cursor
        )
        guard formatVersion == 1 || formatVersion == 2 || formatVersion == mmapVersion else {
            throw ANNSError.corruptFile("Unsupported file version \(formatVersion)")
        }

        let nodeCount = Int(try readUInt32(from: UnsafeRawPointer(region.pointer), length: region.length, cursor: &cursor))
        let degree = Int(try readUInt32(from: UnsafeRawPointer(region.pointer), length: region.length, cursor: &cursor))
        let dim = Int(try readUInt32(from: UnsafeRawPointer(region.pointer), length: region.length, cursor: &cursor))
        let metricCode = try readUInt32(from: UnsafeRawPointer(region.pointer), length: region.length, cursor: &cursor)
        let metric = try metric(from: metricCode)

        let storageType: UInt32 = if formatVersion >= 2 {
            try readUInt32(from: UnsafeRawPointer(region.pointer), length: region.length, cursor: &cursor)
        } else {
            0
        }

        guard nodeCount > 0 else {
            throw ANNSError.corruptFile("Node count must be greater than zero")
        }
        guard degree > 0 else {
            throw ANNSError.corruptFile("Degree must be greater than zero")
        }
        guard dim > 0 else {
            throw ANNSError.corruptFile("Dimension must be greater than zero")
        }
        guard storageType == 0 || storageType == 1 || storageType == 2 else {
            throw ANNSError.corruptFile("Unsupported storage type \(storageType)")
        }
        let isBinary = storageType == 2
        let vectorByteCount: Int
        if isBinary {
            guard dim % 8 == 0 else {
                throw ANNSError.corruptFile("Binary index has dim not divisible by 8")
            }
            vectorByteCount = try checkedMultiply(nodeCount, dim / 8)
        } else {
            let bytesPerElement = storageType == 1 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
            let vectorElements = try checkedMultiply(nodeCount, dim)
            vectorByteCount = try checkedMultiply(vectorElements, bytesPerElement)
        }

        let edgeElements = try checkedMultiply(nodeCount, degree)
        let adjacencyByteCount = try checkedMultiply(edgeElements, MemoryLayout<UInt32>.stride)
        let distanceByteCount = try checkedMultiply(edgeElements, MemoryLayout<Float>.stride)

        let vectorOffset: Int
        let adjacencyOffset: Int
        let distanceOffset: Int
        let trailerOffset: Int

        if formatVersion == mmapVersion {
            vectorOffset = alignedOffset(cursor, alignment: pageSize)
            adjacencyOffset = alignedOffset(try checkedAdd(vectorOffset, vectorByteCount), alignment: pageSize)
            distanceOffset = alignedOffset(try checkedAdd(adjacencyOffset, adjacencyByteCount), alignment: pageSize)
            trailerOffset = alignedOffset(try checkedAdd(distanceOffset, distanceByteCount), alignment: pageSize)
        } else {
            vectorOffset = cursor
            adjacencyOffset = try checkedAdd(vectorOffset, vectorByteCount)
            distanceOffset = try checkedAdd(adjacencyOffset, adjacencyByteCount)
            trailerOffset = try checkedAdd(distanceOffset, distanceByteCount)
        }

        let vectorEnd = try checkedAdd(vectorOffset, vectorByteCount)
        let adjacencyEnd = try checkedAdd(adjacencyOffset, adjacencyByteCount)
        let distanceEnd = try checkedAdd(distanceOffset, distanceByteCount)

        guard vectorEnd <= region.length,
              adjacencyEnd <= region.length,
              distanceEnd <= region.length,
              trailerOffset + MemoryLayout<UInt32>.size <= region.length else {
            throw ANNSError.corruptFile("Index payload is truncated")
        }

        let vectors = try DiskBackedVectorBuffer(
            mmapPointer: UnsafeRawPointer(region.pointer),
            dataOffset: vectorOffset,
            dim: dim,
            count: nodeCount,
            isFloat16: storageType == 1,
            isBinary: isBinary,
            device: metalDevice
        )

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: metalDevice)
        graph.adjacencyBuffer.contents().copyMemory(
            from: region.pointer.advanced(by: adjacencyOffset),
            byteCount: adjacencyByteCount
        )
        graph.distanceBuffer.contents().copyMemory(
            from: region.pointer.advanced(by: distanceOffset),
            byteCount: distanceByteCount
        )
        graph.setCount(nodeCount)

        var trailerCursor = trailerOffset
        let idMapByteCount = Int(
            try readUInt32(
                from: UnsafeRawPointer(region.pointer),
                length: region.length,
                cursor: &trailerCursor
            )
        )

        let idMapStart = trailerCursor
        let idMapEnd = try checkedAdd(idMapStart, idMapByteCount)
        let entryPointEnd = try checkedAdd(idMapEnd, MemoryLayout<UInt32>.size)
        guard entryPointEnd <= region.length else {
            throw ANNSError.corruptFile("Truncated IDMap payload")
        }

        let idMapData = Data(bytes: region.pointer.advanced(by: idMapStart), count: idMapByteCount)
        trailerCursor = idMapEnd
        let entryPoint = try readUInt32(
            from: UnsafeRawPointer(region.pointer),
            length: region.length,
            cursor: &trailerCursor
        )

        let idMap: IDMap
        do {
            idMap = try JSONDecoder().decode(IDMap.self, from: idMapData)
        } catch {
            throw ANNSError.corruptFile("IDMap payload is corrupt")
        }

        return LoadResult(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: metric,
            isBinary: isBinary,
            mmapLifetime: region
        )
    }

    private static func metric(from code: UInt32) throws -> Metric {
        switch code {
        case 0:
            return .cosine
        case 1:
            return .l2
        case 2:
            return .innerProduct
        case 3:
            return .hamming
        default:
            throw ANNSError.corruptFile("Unsupported metric code")
        }
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        if overflow {
            throw ANNSError.corruptFile("Index payload overflows Int bounds")
        }
        return result
    }

    private static func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        if overflow {
            throw ANNSError.corruptFile("Index payload overflows Int bounds")
        }
        return result
    }

    private static func alignedOffset(_ value: Int, alignment: Int) -> Int {
        let remainder = value % alignment
        if remainder == 0 {
            return value
        }
        return value + (alignment - remainder)
    }

    private static func readBytes(
        from pointer: UnsafeRawPointer,
        length: Int,
        offset: Int,
        count: Int
    ) throws -> [UInt8] {
        guard offset >= 0, count >= 0, offset + count <= length else {
            throw ANNSError.corruptFile("Unexpected EOF")
        }

        let base = pointer.bindMemory(to: UInt8.self, capacity: length)
        return Array(UnsafeBufferPointer(start: base.advanced(by: offset), count: count))
    }

    private static func readUInt32(
        from pointer: UnsafeRawPointer,
        length: Int,
        cursor: inout Int
    ) throws -> UInt32 {
        guard cursor + MemoryLayout<UInt32>.size <= length else {
            throw ANNSError.corruptFile("Unexpected EOF")
        }

        let base = pointer.bindMemory(to: UInt8.self, capacity: length).advanced(by: cursor)
        let value = UInt32(base[0]) |
            (UInt32(base[1]) << 8) |
            (UInt32(base[2]) << 16) |
            (UInt32(base[3]) << 24)
        cursor += MemoryLayout<UInt32>.size
        return value
    }
}

private final class DiskBackedMmapRegion: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    let length: Int
    private let descriptor: Int32

    init(fileURL: URL) throws {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ANNSError.corruptFile("Unable to open disk-backed index file")
        }

        var fileStat = stat()
        guard fstat(fd, &fileStat) == 0 else {
            _ = close(fd)
            throw ANNSError.corruptFile("Unable to stat disk-backed index file")
        }

        let mappedLength = Int(fileStat.st_size)
        guard mappedLength > 0 else {
            _ = close(fd)
            throw ANNSError.corruptFile("Disk-backed index file is empty")
        }

        let mappedPointer = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0)
        guard mappedPointer != MAP_FAILED else {
            _ = close(fd)
            throw ANNSError.corruptFile("mmap failed for disk-backed index file")
        }

        self.pointer = mappedPointer!
        self.length = mappedLength
        self.descriptor = fd
    }

    deinit {
        _ = munmap(pointer, length)
        _ = close(descriptor)
    }
}
