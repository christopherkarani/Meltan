import Foundation
import Metal

/// Abstraction over Float32 and Float16 vector storage.
/// Both buffer types conform so construction, search, and pruning can work generically.
package protocol VectorStorage: AnyObject, Sendable {
    var buffer: MTLBuffer { get }
    var dim: Int { get }
    var capacity: Int { get }
    var count: Int { get }
    var isFloat16: Bool { get }

    func setCount(_ newCount: Int)
    func insert(vector: [Float], at index: Int) throws
    func batchInsert(vectors: [[Float]], startingAt start: Int) throws
    func vector(at index: Int) -> [Float]
}
