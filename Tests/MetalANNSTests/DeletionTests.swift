import Testing
@testable import MetalANNSCore

@Suite("Soft Deletion Tests")
struct DeletionTests {
    @Test("Deleted IDs are filtered from results")
    func deletedNotInResults() async throws {
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

        var deletionState = SoftDeletion()
        deletionState.markDeleted(0)
        deletionState.markDeleted(5)
        deletionState.markDeleted(10)

        for queryIndex in 0..<10 {
            let query = vectors[queryIndex]
            let results = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graphData,
                entryPoint: Int(entryPoint),
                k: 10,
                ef: 64,
                metric: .cosine
            )

            let filtered = deletionState.filterResults(results)
            #expect(!filtered.contains(where: { $0.internalID == 0 }))
            #expect(!filtered.contains(where: { $0.internalID == 5 }))
            #expect(!filtered.contains(where: { $0.internalID == 10 }))
        }
    }

    @Test("Deleted count tracks unique IDs")
    func deletedCountTracking() {
        var deletionState = SoftDeletion()

        deletionState.markDeleted(3)
        deletionState.markDeleted(7)
        deletionState.markDeleted(11)
        deletionState.markDeleted(3)
        deletionState.markDeleted(11)

        #expect(deletionState.deletedCount == 3)
    }

    @Test("Undelete restores ID access")
    func undeleteRestores() {
        var deletionState = SoftDeletion()
        deletionState.markDeleted(42)

        #expect(deletionState.isDeleted(42))

        deletionState.undelete(42)
        #expect(!deletionState.isDeleted(42))
    }
}
