import Foundation

public enum ANNSError: Error, Sendable {
    case deviceNotSupported
    case dimensionMismatch(expected: Int, got: Int)
    case idAlreadyExists(String)
    case idNotFound(String)
    case corruptFile(String)
    case constructionFailed(String)
    case capacityExceeded(capacity: Int)
    case searchFailed(String)
    case indexEmpty
}
