import Foundation
import Testing

@testable import MetalANNSCore

@Suite("QuantizedStorage Tests")
struct QuantizedStorageTests {
    private struct MockQuantizedStorage: QuantizedStorage, Codable, Sendable {
        let vectors: [[Float]]

        var count: Int { vectors.count }
        var originalDimension: Int { vectors.first?.count ?? 0 }

        func approximateDistance(query: [Float], to index: UInt32, metric: Metric) -> Float {
            guard Int(index) < vectors.count else {
                return .greatestFiniteMagnitude
            }
            return SIMDDistance.distance(query, vectors[Int(index)], metric: metric)
        }

        func reconstruct(at index: UInt32) -> [Float] {
            guard Int(index) < vectors.count else {
                return []
            }
            // Small, deterministic perturbation to simulate quantization loss.
            return vectors[Int(index)].map { $0 * 0.98 }
        }
    }

    @Test("Protocol can be instantiated via concrete conformance")
    func protocolExists() {
        let storage = MockQuantizedStorage(vectors: [[1, 2, 3], [4, 5, 6]])
        let asProtocol: any QuantizedStorage = storage

        #expect(asProtocol.count == 2)
        #expect(asProtocol.originalDimension == 3)
    }

    @Test("Reconstruction error remains below 5% of source norm")
    func reconstructionError() {
        let vector: [Float] = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9]
        let storage = MockQuantizedStorage(vectors: [vector])
        let reconstructed = storage.reconstruct(at: 0)

        let sourceNorm = vector.reduce(Float(0)) { $0 + ($1 * $1) }.squareRoot()
        let diffNorm = zip(vector, reconstructed).reduce(Float(0)) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + (delta * delta)
        }.squareRoot()
        #expect(diffNorm / sourceNorm < 0.05)
    }

    @Test("Codable round-trip preserves quantized payload")
    func codableRoundTrip() throws {
        let original = MockQuantizedStorage(vectors: [[1, 2], [3, 4], [5, 6]])
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(MockQuantizedStorage.self, from: data)

        #expect(decoded.count == original.count)
        #expect(decoded.originalDimension == original.originalDimension)
        #expect(decoded.vectors == original.vectors)
    }
}
