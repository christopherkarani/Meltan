import Foundation

/// Deterministic, CPU-only synthetic data shared by tests and benchmarks.
///
/// The module is dev-only: it is intentionally absent from the package's
/// library products and must never ship to consumers of MetalANNS.
public enum Fixtures {
    /// Generates `count` vectors of dimension `dim` from the long-standing
    /// deterministic family `sin(i * 0.173) + cos(i * 0.071)`, where
    /// `i = (row + seedOffset) * dim + col`. The frequencies are parameters so
    /// historical variants can be reproduced; the defaults match every current
    /// call site.
    public static func syntheticVectors(
        count: Int,
        dim: Int,
        seedOffset: Int,
        sinFrequency: Float = 0.173,
        cosFrequency: Float = 0.071
    ) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * sinFrequency) + cos(i * cosFrequency)
            }
        }
    }
}
