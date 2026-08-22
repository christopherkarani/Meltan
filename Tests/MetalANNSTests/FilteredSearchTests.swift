import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Filtered Search Tests")
struct FilteredSearchTests {
    @Test("Filtered search returns only matching results")
    func filteredSearchReturnsOnlyMatching() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<100 {
            try await index.setMetadata("category", value: i < 50 ? "A" : "B", for: "v\(i)")
        }

        let results = try await index.search(
            query: vectors[10],
            k: 20,
            filter: .equals(column: "category", value: "A")
        )

        #expect(!results.isEmpty)
        for result in results {
            #expect(extractIndex(from: result.id) < 50)
        }
    }

    @Test("Filtered search maintains reasonable recall")
    func filteredSearchRecall() async throws {
        let dim = 32
        let vectors = makeVectors(count: 200, dim: dim, seedOffset: 7)
        let ids = (0..<200).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<200 {
            let category = i.isMultiple(of: 2) ? "even" : "odd"
            try await index.setMetadata("category", value: category, for: "v\(i)")
        }

        let queryIndices = stride(from: 0, to: 80, by: 2).map { $0 }
        var hits = 0

        for idx in queryIndices {
            let results = try await index.search(
                query: vectors[idx],
                k: 5,
                filter: .equals(column: "category", value: "even")
            )
            if results.contains(where: { $0.id == "v\(idx)" }) {
                hits += 1
            }
        }

        let recall = Float(hits) / Float(queryIndices.count)
        #expect(recall >= 0.50)
    }

    @Test("Compound filters work correctly")
    func compoundFilterWorks() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 21)
        let ids = (0..<100).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 96))
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<100 {
            let category = i < 50 ? "A" : "B"
            let score: Float = (i % 50) < 25 ? 1.0 : 5.0
            try await index.setMetadata("category", value: category, for: "v\(i)")
            try await index.setMetadata("score", value: score, for: "v\(i)")
        }

        let andFilter: Advanced.LegacyFilter = .and([
            .equals(column: "category", value: "A"),
            .greaterThan(column: "score", value: 3.0),
        ])
        let andResults = try await index.search(query: vectors[30], k: 20, filter: andFilter)
        #expect(!andResults.isEmpty)
        for result in andResults {
            let i = extractIndex(from: result.id)
            #expect(i >= 25 && i < 50)
        }

        let orFilter: Advanced.LegacyFilter = .or([
            .equals(column: "category", value: "A"),
            .greaterThan(column: "score", value: 3.0),
        ])
        let orResults = try await index.search(query: vectors[80], k: 30, filter: orFilter)
        #expect(!orResults.isEmpty)
        for result in orResults {
            let i = extractIndex(from: result.id)
            let allowed = (0..<50).contains(i) || (75..<100).contains(i)
            #expect(allowed)
        }

        let notFilter: Advanced.LegacyFilter = .not(.equals(column: "category", value: "A"))
        let notResults = try await index.search(query: vectors[60], k: 20, filter: notFilter)
        #expect(!notResults.isEmpty)
        for result in notResults {
            #expect(extractIndex(from: result.id) >= 50)
        }
    }

    @Test("Int64 filters preserve precision for large values")
    func int64FilterPrecision() async throws {
        let dim = 8
        let vectors = makeVectors(count: 20, dim: dim, seedOffset: 200)
        let ids = (0..<20).map { "v\($0)" }

        let index = GraphIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine, efSearch: 64))
        try await index.build(vectors: vectors, ids: ids)

        let base: Int64 = 9_000_000_000_000_000_000
        for i in 0..<20 {
            try await index.setMetadata("ts", value: base + Int64(i), for: "v\(i)")
        }

        let gtResults = try await index.search(
            query: vectors[10],
            k: 20,
            filter: .greaterThanInt(column: "ts", value: base + 15)
        )
        #expect(!gtResults.isEmpty)
        for result in gtResults {
            let i = extractIndex(from: result.id)
            #expect(i > 15)
        }

        let ltResults = try await index.search(
            query: vectors[10],
            k: 20,
            filter: .lessThanInt(column: "ts", value: base + 5)
        )
        #expect(!ltResults.isEmpty)
        for result in ltResults {
            let i = extractIndex(from: result.id)
            #expect(i < 5)
        }
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.113) + cos(i * 0.071)
            }
        }
    }

    private func extractIndex(from id: String) -> Int {
        Int(id.dropFirst()) ?? -1
    }
}
