import Foundation

public protocol ComputeBackend: Sendable {
    func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float]
}

public enum BackendFactory {
    public static func makeBackend() throws -> any ComputeBackend {
        #if targetEnvironment(simulator)
            return AccelerateBackend()
        #else
            if let metalBackend = try? MetalBackend() {
                return metalBackend
            }
            return AccelerateBackend()
        #endif
    }
}
