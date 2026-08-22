import Foundation
import MetalANNSCore

public protocol IndexKey: Hashable, Sendable, LosslessStringConvertible {}
extension String: IndexKey {}
extension UInt64: IndexKey {}

public enum VectorIndexState {
    public enum Unbuilt: Sendable {}
    public enum Ready: Sendable {}
    public enum ReadOnly: Sendable {}
}

public enum ReadOnlyLoadMode: Sendable {
    case mmap
    case diskBacked
}

public struct VectorRecord<Key: IndexKey>: Sendable {
    public let id: Key
    public let vector: [Float]

    public init(id: Key, vector: [Float]) {
        self.id = id
        self.vector = vector
    }
}

public struct VectorNeighbor<Key: IndexKey>: Sendable {
    public let id: Key
    public let score: Float
    public let internalID: UInt32

    public init(id: Key, score: Float, internalID: UInt32) {
        self.id = id
        self.score = score
        self.internalID = internalID
    }
}

public struct Field<Value>: Sendable {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }
}

public indirect enum QueryFilter: Sendable {
    case any
    case equals(column: String, value: String)
    case greaterThan(column: String, value: Float)
    case lessThan(column: String, value: Float)
    case greaterThanInt(column: String, value: Int64)
    case lessThanInt(column: String, value: Int64)
    case `in`(column: String, values: Set<String>)
    case and([QueryFilter])
    case or([QueryFilter])
    case not(QueryFilter)

    public static func equals(_ field: Field<String>, _ value: String) -> QueryFilter {
        .equals(column: field.name, value: value)
    }

    public static func greaterThan(_ field: Field<Float>, _ value: Float) -> QueryFilter {
        .greaterThan(column: field.name, value: value)
    }

    public static func lessThan(_ field: Field<Float>, _ value: Float) -> QueryFilter {
        .lessThan(column: field.name, value: value)
    }

    public static func greaterThan(_ field: Field<Int64>, _ value: Int64) -> QueryFilter {
        .greaterThanInt(column: field.name, value: value)
    }

    public static func lessThan(_ field: Field<Int64>, _ value: Int64) -> QueryFilter {
        .lessThanInt(column: field.name, value: value)
    }

    public static func oneOf(_ field: Field<String>, _ values: Set<String>) -> QueryFilter {
        .in(column: field.name, values: values)
    }

    fileprivate func toCoreFilter() -> _LegacySearchFilter? {
        switch self {
        case .any:
            return nil
        case .equals(let column, let value):
            return .equals(column: column, value: value)
        case .greaterThan(let column, let value):
            return .greaterThan(column: column, value: value)
        case .lessThan(let column, let value):
            return .lessThan(column: column, value: value)
        case .greaterThanInt(let column, let value):
            return .greaterThanInt(column: column, value: value)
        case .lessThanInt(let column, let value):
            return .lessThanInt(column: column, value: value)
        case .in(let column, let values):
            return .in(column: column, values: values)
        case .and(let filters):
            return .and(filters.compactMap { $0.toCoreFilter() })
        case .or(let filters):
            return .or(filters.compactMap { $0.toCoreFilter() })
        case .not(let filter):
            if let core = filter.toCoreFilter() {
                return .not(core)
            }
            return nil
        }
    }
}

@resultBuilder
public enum QueryFilterBuilder {
    public static func buildBlock(_ components: QueryFilter...) -> QueryFilter {
        collapse(components)
    }

    public static func buildOptional(_ component: QueryFilter?) -> QueryFilter {
        component ?? .any
    }

    public static func buildEither(first component: QueryFilter) -> QueryFilter {
        component
    }

    public static func buildEither(second component: QueryFilter) -> QueryFilter {
        component
    }

    public static func buildArray(_ components: [QueryFilter]) -> QueryFilter {
        collapse(components)
    }

    private static func collapse(_ components: [QueryFilter]) -> QueryFilter {
        let filtered = components.filter {
            if case .any = $0 {
                return false
            }
            return true
        }
        switch filtered.count {
        case 0: return .any
        case 1: return filtered[0]
        default: return .and(filtered)
        }
    }
}

public struct VectorIndex<Key: IndexKey, State>: Sendable {
    private let rawIndex: GraphIndex

    private init(rawIndex: GraphIndex) {
        self.rawIndex = rawIndex
    }

    /// Power-user escape hatch that exposes the underlying `GraphIndex` actor.
    ///
    /// > Warning: Accessing `advanced` bypasses the phantom-type state machine.
    /// > The caller is responsible for ensuring operations are valid for the
    /// > current index state (e.g., do not call `search` on an unbuilt index).
    /// > Prefer the typed API (`build`, `search`, `insert`, etc.) whenever possible.
    public var advanced: GraphIndex {
        rawIndex
    }

    fileprivate static func mapNeighbors(_ results: [SearchResult]) -> [VectorNeighbor<Key>] {
        results.compactMap { result in
            let candidate: Key? =
                if !result.id.isEmpty {
                    Key(result.id)
                } else if let numericID = result.numericID {
                    Key(String(numericID))
                } else {
                    nil
                }

            guard let id = candidate else {
                return nil
            }
            return VectorNeighbor(id: id, score: result.score, internalID: result.internalID)
        }
    }
}

extension VectorIndex where State == VectorIndexState.Unbuilt {
    public init(configuration: IndexConfiguration = .default) {
        self.init(rawIndex: GraphIndex(configuration: configuration))
    }

    public consuming func build(records: [VectorRecord<Key>]) async throws -> VectorIndex<Key, VectorIndexState.Ready> {
        guard !records.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build index with empty records")
        }
        let vectors = records.map(\.vector)
        let ids = records.map { String($0.id) }
        try await rawIndex.build(vectors: vectors, ids: ids)
        return VectorIndex<Key, VectorIndexState.Ready>(rawIndex: rawIndex)
    }

    public consuming func build(vectors: [[Float]], ids: [Key]) async throws -> VectorIndex<Key, VectorIndexState.Ready>
    {
        try await build(records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) })
    }

    public static func load(from url: URL) async throws -> VectorIndex<Key, VectorIndexState.Ready> {
        let loaded = try await GraphIndex.load(from: url)
        return VectorIndex<Key, VectorIndexState.Ready>(rawIndex: loaded)
    }

    public static func loadReadOnly(
        from url: URL,
        mode: ReadOnlyLoadMode = .mmap
    ) async throws -> VectorIndex<Key, VectorIndexState.ReadOnly> {
        let loaded: GraphIndex
        switch mode {
        case .mmap:
            loaded = try await GraphIndex.loadMmap(from: url)
        case .diskBacked:
            loaded = try await GraphIndex.loadDiskBacked(from: url)
        }
        return VectorIndex<Key, VectorIndexState.ReadOnly>(rawIndex: loaded)
    }
}

extension VectorIndex where State == VectorIndexState.Ready {
    public var count: Int {
        get async {
            await rawIndex.count
        }
    }

    public func insert(_ record: VectorRecord<Key>) async throws {
        try await rawIndex.insert(record.vector, id: String(record.id))
    }

    public func batchInsert(_ records: [VectorRecord<Key>]) async throws {
        let vectors = records.map(\.vector)
        let ids = records.map { String($0.id) }
        try await rawIndex.batchInsert(vectors, ids: ids)
    }

    public func delete(id: Key) async throws {
        try await rawIndex.delete(id: String(id))
    }

    public func compact() async throws {
        try await rawIndex.compact()
    }

    public func save(to url: URL) async throws {
        try await rawIndex.save(to: url)
    }

    public func saveMmapCompatible(to url: URL) async throws {
        try await rawIndex.saveMmapCompatible(to: url)
    }

    public func setMetadata(_ field: Field<String>, value: String, for id: Key) async throws {
        try await rawIndex.setMetadata(field.name, value: value, for: String(id))
    }

    public func setMetadata(_ field: Field<Float>, value: Float, for id: Key) async throws {
        try await rawIndex.setMetadata(field.name, value: value, for: String(id))
    }

    public func setMetadata(_ field: Field<Int64>, value: Int64, for id: Key) async throws {
        try await rawIndex.setMetadata(field.name, value: value, for: String(id))
    }

    public func search(
        query: [Float],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.search(
            query: query,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results)
    }

    public func batchSearch(
        queries: [[Float]],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [[VectorNeighbor<Key>]] {
        let results = try await rawIndex.batchSearch(
            queries: queries,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return results.map(Self.mapNeighbors)
    }

    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.rangeSearch(
            query: query,
            maxDistance: maxDistance,
            limit: limit,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results)
    }
}

extension VectorIndex where State == VectorIndexState.ReadOnly {
    public var count: Int {
        get async {
            await rawIndex.count
        }
    }

    public func save(to url: URL) async throws {
        try await rawIndex.save(to: url)
    }

    public func search(
        query: [Float],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.search(
            query: query,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results)
    }

    public func batchSearch(
        queries: [[Float]],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [[VectorNeighbor<Key>]] {
        let results = try await rawIndex.batchSearch(
            queries: queries,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return results.map(Self.mapNeighbors)
    }

    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.rangeSearch(
            query: query,
            maxDistance: maxDistance,
            limit: limit,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results)
    }
}
