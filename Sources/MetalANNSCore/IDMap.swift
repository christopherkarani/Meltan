import Foundation

/// Bidirectional mapping between external String IDs and internal UInt32 IDs.
public struct IDMap: Sendable, Codable {
    private var externalToInternal: [String: UInt32] = [:]
    private var internalToExternal: [UInt32: String] = [:]
    private var nextID: UInt32 = 0
    private var numericToInternal: [UInt64: UInt32] = [:]
    private var internalToNumeric: [UInt32: UInt64] = [:]

    public init() {}

    /// The next internal ID that will be assigned.
    public var nextInternalID: UInt32 {
        nextID
    }

    /// Returns true if at least `count` additional IDs can be assigned.
    public func canAllocate(_ count: Int) -> Bool {
        guard count >= 0 else {
            return false
        }
        // Reserve UInt32.max as invalid/sentinel in graph structures.
        let remaining = Int(UInt32.max &- nextID)
        return count <= remaining
    }

    /// Assigns a new internal ID. Returns nil if the external ID already exists.
    public mutating func assign(externalID: String) -> UInt32? {
        guard externalToInternal[externalID] == nil else {
            return nil
        }
        guard nextID < UInt32.max else {
            return nil
        }

        let internalID = nextID
        externalToInternal[externalID] = internalID
        internalToExternal[internalID] = externalID
        nextID += 1
        return internalID
    }

    public func internalID(for externalID: String) -> UInt32? {
        externalToInternal[externalID]
    }

    public func externalID(for internalID: UInt32) -> String? {
        internalToExternal[internalID]
    }

    public var count: Int {
        externalToInternal.count + numericToInternal.count
    }
}

// MARK: - UInt64 Key Support

extension IDMap {
    /// Assigns a new internal ID for a numeric (UInt64) key.
    /// Returns nil if the numeric key already exists or capacity is exhausted.
    public mutating func assign(numericID: UInt64) -> UInt32? {
        guard numericToInternal[numericID] == nil else {
            return nil
        }
        guard nextID < UInt32.max else {
            return nil
        }

        let internalID = nextID
        numericToInternal[numericID] = internalID
        internalToNumeric[internalID] = numericID
        nextID += 1
        return internalID
    }

    public func internalID(forNumeric numericID: UInt64) -> UInt32? {
        numericToInternal[numericID]
    }

    public func numericID(for internalID: UInt32) -> UInt64? {
        internalToNumeric[internalID]
    }
}

// MARK: - Codable (explicit, for backward-compat when loading old indexes)

extension IDMap {
    private enum CodingKeys: CodingKey {
        case externalToInternal
        case internalToExternal
        case nextID
        case numericToInternal
        case internalToNumeric
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        externalToInternal = try c.decode([String: UInt32].self, forKey: .externalToInternal)
        internalToExternal = try c.decode([UInt32: String].self, forKey: .internalToExternal)
        nextID = try c.decode(UInt32.self, forKey: .nextID)
        // Backward compatibility: older indexes do not include numeric maps.
        numericToInternal = try c.decodeIfPresent([UInt64: UInt32].self, forKey: .numericToInternal) ?? [:]
        internalToNumeric = try c.decodeIfPresent([UInt32: UInt64].self, forKey: .internalToNumeric) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(externalToInternal, forKey: .externalToInternal)
        try c.encode(internalToExternal, forKey: .internalToExternal)
        try c.encode(nextID, forKey: .nextID)
        try c.encode(numericToInternal, forKey: .numericToInternal)
        try c.encode(internalToNumeric, forKey: .internalToNumeric)
    }
}
