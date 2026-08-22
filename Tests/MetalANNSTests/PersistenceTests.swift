import Foundation
import Testing

@testable import MetalANNSCore

@Suite("Persistence Tests")
struct PersistenceTests {
    @Test("Save and load roundtrip")
    func saveAndLoadRoundtrip() async throws {
        let n = 50
        let dim = 8
        let degree = 4

        let vectors = seededVectors(count: n, dim: dim)
        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: 10
        )

        let vectorBuffer = try buildVectorBuffer(from: vectors)
        let graphBuffer = try buildGraphBuffer(from: graphData, degree: degree, capacity: n)

        var idMap = IDMap()
        for index in 0..<n {
            _ = idMap.assign(externalID: "node-\(index)")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-persistence-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try IndexSerializer.save(
            vectors: vectorBuffer,
            graph: graphBuffer,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: .cosine,
            to: tempURL
        )

        let loaded = try IndexSerializer.load(from: tempURL)

        #expect(loaded.vectors.dim == dim)
        #expect(loaded.graph.degree == degree)
        #expect(loaded.vectors.count == n)
        #expect(loaded.graph.nodeCount == n)
        #expect(loaded.entryPoint == entryPoint)
        #expect(loaded.idMap.count == n)
        #expect(loaded.idMap.internalID(for: "node-\(Int(entryPoint))") == entryPoint)
        #expect(loaded.metric == .cosine)

        let loadedVectors = vectorsFromBuffer(loaded.vectors)
        let loadedGraph = graphFromBuffer(loaded.graph)

        for queryIndex in 0..<5 {
            let query = vectors[queryIndex]
            let originalResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graphData,
                entryPoint: Int(entryPoint),
                k: 5,
                ef: 64,
                metric: .cosine
            )

            let loadedResults = try await BeamSearchCPU.search(
                query: query,
                vectors: loadedVectors,
                graph: loadedGraph,
                entryPoint: Int(loaded.entryPoint),
                k: 5,
                ef: 64,
                metric: loaded.metric
            )

            #expect(originalResults.count == loadedResults.count)
            #expect(originalResults.map(\.internalID) == loadedResults.map(\.internalID))
        }
    }

    @Test("Corrupt magic throws")
    func corruptMagicThrows() throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-corrupt-magic-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
        }

        let badPayload: [UInt8] = [0x58, 0x58, 0x58, 0x58]  // "XXXX"
        FileManager.default.createFile(atPath: fileURL.path, contents: Data(badPayload))

        do {
            _ = try IndexSerializer.load(from: fileURL)
            #expect(Bool(false), "Expected corrupt file error")
        } catch let error as ANNSError {
            if case .corruptFile = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.corruptFile but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.corruptFile")
        }
    }

    @Test("Corrupt version throws")
    func corruptVersionThrows() throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-corrupt-version-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
        }

        var payload = Data()
        payload.append(0x4D)
        payload.append(0x41)
        payload.append(0x4E)
        payload.append(0x4E)
        appendUInt32(99, to: &payload)
        appendUInt32(1, to: &payload)
        appendUInt32(1, to: &payload)
        appendUInt32(1, to: &payload)
        appendUInt32(0, to: &payload)

        FileManager.default.createFile(atPath: fileURL.path, contents: payload)

        do {
            _ = try IndexSerializer.load(from: fileURL)
            #expect(Bool(false), "Expected corrupt file error")
        } catch let error as ANNSError {
            if case .corruptFile = error {
            } else {
                #expect(Bool(false), "Expected ANNSError.corruptFile but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.corruptFile")
        }
    }

    private func buildVectorBuffer(from vectors: [[Float]]) throws -> VectorBuffer {
        guard let first = vectors.first else {
            throw ANNSError.constructionFailed("Vector list cannot be empty")
        }

        let buffer = try VectorBuffer(capacity: vectors.count, dim: first.count)
        for (index, vector) in vectors.enumerated() {
            try buffer.insert(vector: vector, at: index)
        }
        buffer.setCount(vectors.count)
        return buffer
    }

    private func buildGraphBuffer(from graphData: [[(UInt32, Float)]], degree: Int, capacity: Int) throws -> GraphBuffer
    {
        let graphBuffer = try GraphBuffer(capacity: capacity, degree: degree)
        for node in 0..<graphData.count {
            let neighbors = graphData[node]
            var neighborIDs = Array(repeating: UInt32.max, count: degree)
            var neighborDistances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)

            for index in 0..<min(degree, neighbors.count) {
                neighborIDs[index] = neighbors[index].0
                neighborDistances[index] = neighbors[index].1
            }

            try graphBuffer.setNeighbors(of: node, ids: neighborIDs, distances: neighborDistances)
        }
        graphBuffer.setCount(graphData.count)
        return graphBuffer
    }

    private func vectorsFromBuffer(_ vectorBuffer: any VectorStorage) -> [[Float]] {
        var vectors = [[Float]]()
        vectors.reserveCapacity(vectorBuffer.count)

        for index in 0..<vectorBuffer.count {
            let row = vectorBuffer.vector(at: index)
            vectors.append(row)
        }
        return vectors
    }

    private func graphFromBuffer(_ graph: GraphBuffer) -> [[(UInt32, Float)]] {
        (0..<graph.nodeCount).map { node in
            let ids = graph.neighborIDs(of: node)
            let distances = graph.neighborDistances(of: node)
            return Array(zip(ids, distances))
        }
    }
}

private func appendUInt32(_ value: UInt32, to payload: inout Data) {
    var littleEndianValue = value.littleEndian
    payload.append(Data(bytes: &littleEndianValue, count: MemoryLayout<UInt32>.size))
}
