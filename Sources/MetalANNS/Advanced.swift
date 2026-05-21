import Foundation
import MetalANNSCore

/// Explicit power-user surface for low-level control.
///
/// > Note: `GraphIndex` is now a top-level public type. Use `GraphIndex` directly
/// > instead of `GraphIndex`.
public enum Advanced {
    public typealias StreamingIndex = MetalANNS._StreamingIndex
    public typealias ShardedIndex = MetalANNS._ShardedIndex
    public typealias IVFPQIndex = MetalANNS._IVFPQIndex
    public typealias LegacyFilter = _LegacySearchFilter
}
