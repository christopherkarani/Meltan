import Darwin
import Foundation
import Metal

package enum IndexSerializer {
    private static let headerMagic: [UInt8] = [0x4D, 0x41, 0x4E, 0x4E]  // "MANN"
    private static let version: UInt32 = 2
    private static let mmapVersion: UInt32 = 3
    private static var pageSize: Int { max(1, Int(getpagesize())) }

    package static func save(
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        metric: Metric,
        to fileURL: URL
    ) throws {
        let nodeCount = vectors.count
        let degree = graph.degree

        guard nodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot save empty index")
        }
        guard nodeCount == graph.nodeCount else {
            throw ANNSError.constructionFailed("Vector and graph node counts do not match")
        }
        guard nodeCount <= Int(UInt32.max),
            degree <= Int(UInt32.max),
            vectors.dim <= Int(UInt32.max)
        else {
            throw ANNSError.constructionFailed("Index dimensions exceed supported serialization limits")
        }

        let storageType: UInt32
        let vectorByteCount: Int
        if let binaryBuffer = vectors as? BinaryVectorBuffer {
            storageType = 2
            vectorByteCount = try checkedMultiply(nodeCount, binaryBuffer.bytesPerVector)
        } else {
            storageType = vectors.isFloat16 ? 1 : 0
            let bytesPerElement = vectors.isFloat16 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
            let vectorElementCount = try checkedMultiply(nodeCount, vectors.dim)
            vectorByteCount = try checkedMultiply(vectorElementCount, bytesPerElement)
        }
        let edgeCount = try checkedMultiply(nodeCount, degree)
        let adjacencyByteCount = try checkedMultiply(edgeCount, MemoryLayout<UInt32>.stride)
        let distanceByteCount = try checkedMultiply(edgeCount, MemoryLayout<Float>.stride)

        var filePayload = Data()
        append(magic: headerMagic, to: &filePayload)
        append(uint32: version, to: &filePayload)
        append(uint32: UInt32(nodeCount), to: &filePayload)
        append(uint32: UInt32(degree), to: &filePayload)
        append(uint32: UInt32(vectors.dim), to: &filePayload)
        append(uint32: metricCode(metric), to: &filePayload)
        append(uint32: storageType, to: &filePayload)

        filePayload.append(Data(bytes: vectors.buffer.contents(), count: vectorByteCount))
        filePayload.append(Data(bytes: graph.adjacencyBuffer.contents(), count: adjacencyByteCount))
        filePayload.append(Data(bytes: graph.distanceBuffer.contents(), count: distanceByteCount))

        let idMapData = try JSONEncoder().encode(idMap)
        guard idMapData.count <= Int(UInt32.max) else {
            throw ANNSError.constructionFailed("ID map payload exceeds supported size")
        }
        append(uint32: UInt32(idMapData.count), to: &filePayload)
        filePayload.append(idMapData)
        append(uint32: entryPoint, to: &filePayload)

        try atomicWrite(filePayload, to: fileURL)
    }

    package static func saveMmapCompatible(
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        metric: Metric,
        to fileURL: URL
    ) throws {
        let nodeCount = vectors.count
        let degree = graph.degree

        guard nodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot save empty index")
        }
        guard nodeCount == graph.nodeCount else {
            throw ANNSError.constructionFailed("Vector and graph node counts do not match")
        }

        guard nodeCount <= Int(UInt32.max),
            degree <= Int(UInt32.max),
            vectors.dim <= Int(UInt32.max)
        else {
            throw ANNSError.constructionFailed("Index dimensions exceed supported serialization limits")
        }

        let storageType: UInt32
        let vectorByteCount: Int
        if let binaryBuffer = vectors as? BinaryVectorBuffer {
            storageType = 2
            vectorByteCount = try checkedMultiply(nodeCount, binaryBuffer.bytesPerVector)
        } else {
            storageType = vectors.isFloat16 ? 1 : 0
            let bytesPerElement = vectors.isFloat16 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
            let vectorElementCount = try checkedMultiply(nodeCount, vectors.dim)
            vectorByteCount = try checkedMultiply(vectorElementCount, bytesPerElement)
        }
        let edgeCount = try checkedMultiply(nodeCount, degree)
        let adjacencyByteCount = try checkedMultiply(edgeCount, MemoryLayout<UInt32>.stride)
        let distanceByteCount = try checkedMultiply(edgeCount, MemoryLayout<Float>.stride)

        var filePayload = Data()
        append(magic: headerMagic, to: &filePayload)
        append(uint32: mmapVersion, to: &filePayload)
        append(uint32: UInt32(nodeCount), to: &filePayload)
        append(uint32: UInt32(degree), to: &filePayload)
        append(uint32: UInt32(vectors.dim), to: &filePayload)
        append(uint32: metricCode(metric), to: &filePayload)
        append(uint32: storageType, to: &filePayload)

        appendPagePadding(to: &filePayload)
        filePayload.append(Data(bytes: vectors.buffer.contents(), count: vectorByteCount))
        appendPagePadding(to: &filePayload)
        filePayload.append(Data(bytes: graph.adjacencyBuffer.contents(), count: adjacencyByteCount))
        appendPagePadding(to: &filePayload)
        filePayload.append(Data(bytes: graph.distanceBuffer.contents(), count: distanceByteCount))
        appendPagePadding(to: &filePayload)

        let idMapData = try JSONEncoder().encode(idMap)
        guard idMapData.count <= Int(UInt32.max) else {
            throw ANNSError.constructionFailed("ID map payload exceeds supported size")
        }
        append(uint32: UInt32(idMapData.count), to: &filePayload)
        filePayload.append(idMapData)
        append(uint32: entryPoint, to: &filePayload)

        try atomicWrite(filePayload, to: fileURL)
    }

    package static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> (
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        metric: Metric
    ) {
        let payload = try Data(contentsOf: fileURL)

        guard payload.count >= 24 else {
            throw ANNSError.corruptFile("File is too small")
        }

        var cursor = 0
        let magic = Array(payload[cursor..<cursor + 4])
        cursor += 4
        guard magic == headerMagic else {
            throw ANNSError.corruptFile("Invalid file magic")
        }

        let formatVersion = try readUInt32(payload, &cursor)
        guard formatVersion == 1 || formatVersion == version || formatVersion == mmapVersion else {
            throw ANNSError.corruptFile("Unsupported file version \(formatVersion)")
        }

        let nodeCount = Int(try readUInt32(payload, &cursor))
        let degree = Int(try readUInt32(payload, &cursor))
        let dim = Int(try readUInt32(payload, &cursor))
        let metricCode = try readUInt32(payload, &cursor)
        let metric = try metric(from: metricCode)
        let storageType: UInt32 =
            if formatVersion >= 2 {
                try readUInt32(payload, &cursor)
            } else {
                0
            }
        guard storageType == 0 || storageType == 1 || storageType == 2 else {
            throw ANNSError.corruptFile("Unsupported storage type \(storageType)")
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

        let vectorByteCount: Int
        if storageType == 2 {
            guard dim % 8 == 0 else {
                throw ANNSError.corruptFile("Binary index has dim not divisible by 8")
            }
            vectorByteCount = try checkedMultiply(nodeCount, dim / 8)
        } else {
            let bytesPerElement = storageType == 1 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
            let vectorElementCount = try checkedMultiply(nodeCount, dim)
            vectorByteCount = try checkedMultiply(vectorElementCount, bytesPerElement)
        }
        let edgeCount = try checkedMultiply(nodeCount, degree)
        let adjacencyByteCount = try checkedMultiply(edgeCount, MemoryLayout<UInt32>.stride)
        let distanceByteCount = try checkedMultiply(edgeCount, MemoryLayout<Float>.stride)

        let vectorData: Data.SubSequence
        let adjacencyData: Data.SubSequence
        let distanceData: Data.SubSequence

        if formatVersion == mmapVersion {
            let vectorStart = try alignedOffset(cursor)
            let vectorEnd = try checkedAdd(vectorStart, vectorByteCount)
            guard payload.count >= vectorEnd else {
                throw ANNSError.corruptFile("Corrupt or truncated vector payload")
            }
            vectorData = payload[vectorStart..<vectorEnd]

            let adjacencyStart = try alignedOffset(vectorEnd)
            let adjacencyEnd = try checkedAdd(adjacencyStart, adjacencyByteCount)
            guard payload.count >= adjacencyEnd else {
                throw ANNSError.corruptFile("Corrupt or truncated adjacency payload")
            }
            adjacencyData = payload[adjacencyStart..<adjacencyEnd]

            let distanceStart = try alignedOffset(adjacencyEnd)
            let distanceEnd = try checkedAdd(distanceStart, distanceByteCount)
            guard payload.count >= distanceEnd else {
                throw ANNSError.corruptFile("Corrupt or truncated distance payload")
            }
            distanceData = payload[distanceStart..<distanceEnd]

            cursor = try alignedOffset(distanceEnd)
        } else {
            let expectedBodyLength = try checkedAdd(
                vectorByteCount, try checkedAdd(adjacencyByteCount, distanceByteCount))
            let bodyEnd = try checkedAdd(cursor, expectedBodyLength)
            let minimumTail = try checkedAdd(bodyEnd, MemoryLayout<UInt32>.size)
            guard payload.count >= minimumTail else {
                throw ANNSError.corruptFile("Corrupt or truncated body")
            }

            let vectorEnd = try checkedAdd(cursor, vectorByteCount)
            vectorData = payload[cursor..<vectorEnd]
            cursor = vectorEnd

            let adjacencyEnd = try checkedAdd(cursor, adjacencyByteCount)
            adjacencyData = payload[cursor..<adjacencyEnd]
            cursor = adjacencyEnd

            let distanceEnd = try checkedAdd(cursor, distanceByteCount)
            distanceData = payload[cursor..<distanceEnd]
            cursor = distanceEnd
        }

        let idMapByteCount = Int(try readUInt32(payload, &cursor))
        let idMapEnd = try checkedAdd(cursor, idMapByteCount)
        let entryPointEnd = try checkedAdd(idMapEnd, MemoryLayout<UInt32>.size)
        guard payload.count >= entryPointEnd else {
            throw ANNSError.corruptFile("Truncated or invalid IDMap payload")
        }

        let idMapData = payload[cursor..<idMapEnd]
        cursor = idMapEnd

        let entryPoint = try readUInt32(payload, &cursor)

        if cursor > payload.count {
            throw ANNSError.corruptFile("Malformed payload")
        }

        let mutableCapacity = max(nodeCount + 1, nodeCount * 2)
        let vectors: any VectorStorage
        if storageType == 2 {
            vectors = try BinaryVectorBuffer(capacity: mutableCapacity, dim: dim, device: device)
        } else if storageType == 1 {
            vectors = try Float16VectorBuffer(capacity: mutableCapacity, dim: dim, device: device)
        } else {
            vectors = try VectorBuffer(capacity: mutableCapacity, dim: dim, device: device)
        }
        let graph = try GraphBuffer(capacity: mutableCapacity, degree: degree, device: device)

        vectorData.withUnsafeBytes { raw in
            vectors.buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: vectorByteCount)
        }
        adjacencyData.withUnsafeBytes { raw in
            graph.adjacencyBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: adjacencyByteCount)
        }
        distanceData.withUnsafeBytes { raw in
            graph.distanceBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: distanceByteCount)
        }

        vectors.setCount(nodeCount)
        graph.setCount(nodeCount)

        let idMap: IDMap
        do {
            idMap = try JSONDecoder().decode(IDMap.self, from: idMapData)
        } catch {
            throw ANNSError.corruptFile("IDMap payload is corrupt")
        }
        guard idMap.count == nodeCount else {
            throw ANNSError.corruptFile("ID map size does not match node count")
        }
        guard Int(entryPoint) < nodeCount else {
            throw ANNSError.corruptFile("Entry point is out of bounds")
        }

        return (vectors: vectors, graph: graph, idMap: idMap, entryPoint: entryPoint, metric: metric)
    }

    private static func metricCode(_ metric: Metric) -> UInt32 {
        switch metric {
        case .cosine:
            return 0
        case .l2:
            return 1
        case .innerProduct:
            return 2
        case .hamming:
            return 3
        }
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

    private static func append(uint32 value: UInt32, to payload: inout Data) {
        var littleEndianValue = value.littleEndian
        payload.append(Data(bytes: &littleEndianValue, count: MemoryLayout<UInt32>.size))
    }

    private static func append(magic: [UInt8], to payload: inout Data) {
        payload.append(contentsOf: magic)
    }

    private static func appendPagePadding(to payload: inout Data) {
        let padding = (pageSize - (payload.count % pageSize)) % pageSize
        if padding > 0 {
            payload.append(Data(repeating: 0, count: padding))
        }
    }

    private static func alignedOffset(_ value: Int) throws -> Int {
        let remainder = value % pageSize
        if remainder == 0 {
            return value
        }
        return try checkedAdd(value, pageSize - remainder)
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (product, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        if overflow {
            throw ANNSError.corruptFile("Integer overflow while computing serialized sizes")
        }
        return product
    }

    private static func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        if overflow {
            throw ANNSError.corruptFile("Integer overflow while computing serialized offsets")
        }
        return sum
    }

    private static func atomicWrite(_ data: Data, to fileURL: URL) throws {
        let fileManager = FileManager.default
        let parentURL = fileURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentURL, withIntermediateDirectories: true)

        let tempURL = parentURL.appendingPathComponent(".\(fileURL.lastPathComponent).tmp-\(UUID().uuidString)")
        do {
            try data.write(to: tempURL)

            if fileManager.fileExists(atPath: fileURL.path) {
                _ = try fileManager.replaceItemAt(fileURL, withItemAt: tempURL)
            } else {
                try fileManager.moveItem(at: tempURL, to: fileURL)
            }
        } catch {
            try? fileManager.removeItem(at: tempURL)
            throw error
        }
    }

    private static func readUInt32(_ payload: Data, _ cursor: inout Int) throws -> UInt32 {
        guard cursor + MemoryLayout<UInt32>.size <= payload.count else {
            throw ANNSError.corruptFile("Unexpected EOF")
        }

        let value: UInt32 = payload.withUnsafeBytes { raw in
            let start = raw.baseAddress!.advanced(by: cursor).assumingMemoryBound(to: UInt8.self)
            return UInt32(start[0]) | (UInt32(start[1]) << 8) | (UInt32(start[2]) << 16) | (UInt32(start[3]) << 24)
        }
        cursor += MemoryLayout<UInt32>.size
        return value
    }
}
