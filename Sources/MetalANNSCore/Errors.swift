import Foundation

/// Typed domain errors. Invariant: typed domain errors and `CancellationError`
/// propagate verbatim through every layer; only untyped foreign errors may be
/// enveloped.
public enum ANNSError: Error, Sendable {
    case deviceNotSupported
    case dimensionMismatch(expected: Int, got: Int)
    case idAlreadyExists(String)
    case idNotFound(String)
    case corruptFile(String)
    case constructionFailed(String)
    case searchFailed(String)
    case indexEmpty
    case gpuPipelineUnavailable(String)
    case gpuResourceExhausted(String)
    case gpuExecutionFailed(String)
    case indexCapacityExceeded
}

extension ANNSError {
    /// Errors eligible for degrade-to-next-tier in the search cascade.
    package var isTierDegradeable: Bool {
        switch self {
        case .gpuPipelineUnavailable, .gpuResourceExhausted, .gpuExecutionFailed, .searchFailed:
            true
        default:
            false
        }
    }
}
