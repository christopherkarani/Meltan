import Foundation
import Testing

@testable import MetalANNS

@Suite("Advanced.StreamingIndex Insert Tests")
struct StreamingIndexInsertTests {
    @Test("Insert single vector")
    func insertSingleVector() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 10,
                mergeStrategy: .blocking
            ))

        try await index.insert(makeVector(row: 0, dim: 4), id: "v0")

        #expect(await index.count == 1)
    }

    @Test("Insert beyond single capacity")
    func insertBeyondSingleCapacity() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 10,
                mergeStrategy: .blocking
            ))

        for i in 0..<25 {
            try await index.insert(makeVector(row: i, dim: 4), id: "v\(i)")
        }

        #expect(await index.count == 25)
    }

    @Test("Batch insert")
    func batchInsert() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 20,
                mergeStrategy: .blocking
            ))

        let vectors = (0..<50).map { makeVector(row: $0, dim: 4) }
        let ids = (0..<50).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        #expect(await index.count == 50)
    }

    @Test("Duplicate ID throws")
    func duplicateIDThrows() async throws {
        let index = Advanced.StreamingIndex(
            config: StreamingConfiguration(
                deltaCapacity: 10,
                mergeStrategy: .blocking
            ))
        let vector = makeVector(row: 0, dim: 4)

        try await index.insert(vector, id: "a")
        do {
            try await index.insert(vector, id: "a")
            #expect(Bool(false), "Expected ANNSError.idAlreadyExists")
        } catch let error as ANNSError {
            switch error {
            case .idAlreadyExists(let id):
                #expect(id == "a")
            default:
                #expect(Bool(false), "Expected ANNSError.idAlreadyExists but got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.idAlreadyExists")
        }
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.173) + cos(i * 0.071)
        }
    }
}
