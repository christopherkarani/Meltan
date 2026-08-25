package protocol QuantizedStorage: Sendable {
    /// Total number of quantized vectors available.
    var count: Int { get }

    /// Original unquantized vector dimension.
    var originalDimension: Int { get }

    /// Approximate distance from query to vector at index.
    func approximateDistance(query: [Float], to index: UInt32, metric: Metric) -> Float

    /// Lossy reconstruction for debugging and inspection.
    func reconstruct(at index: UInt32) -> [Float]
}
