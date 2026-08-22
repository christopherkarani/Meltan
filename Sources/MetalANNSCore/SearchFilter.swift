import Foundation

public indirect enum _LegacySearchFilter: Sendable {
    case equals(column: String, value: String)
    case greaterThan(column: String, value: Float)
    case lessThan(column: String, value: Float)
    case greaterThanInt(column: String, value: Int64)
    case lessThanInt(column: String, value: Int64)
    case `in`(column: String, values: Set<String>)
    case and([_LegacySearchFilter])
    case or([_LegacySearchFilter])
    case not(_LegacySearchFilter)
}

extension _LegacySearchFilter {
    /// Evaluates this filter against a row exposed through per-column typed
    /// lookups. This is the single predicate ladder shared by `MetadataStore`
    /// and `_StreamingIndex` pending-record evaluation.
    public func evaluate(
        stringValue: (String) -> String?,
        floatValue: (String) -> Float?,
        intValue: (String) -> Int64?
    ) -> Bool {
        switch self {
        case .equals(let column, let value):
            return stringValue(column) == value

        case .greaterThan(let column, let value):
            if let current = floatValue(column) {
                return current > value
            }
            if let current = intValue(column) {
                return Float(current) > value
            }
            return false

        case .lessThan(let column, let value):
            if let current = floatValue(column) {
                return current < value
            }
            if let current = intValue(column) {
                return Float(current) < value
            }
            return false

        case .greaterThanInt(let column, let value):
            guard let current = intValue(column) else {
                return false
            }
            return current > value

        case .lessThanInt(let column, let value):
            guard let current = intValue(column) else {
                return false
            }
            return current < value

        case .in(let column, let values):
            guard let current = stringValue(column) else {
                return false
            }
            return values.contains(current)

        case .and(let filters):
            return filters.allSatisfy { filter in
                filter.evaluate(
                    stringValue: stringValue,
                    floatValue: floatValue,
                    intValue: intValue
                )
            }

        case .or(let filters):
            return filters.contains { filter in
                filter.evaluate(
                    stringValue: stringValue,
                    floatValue: floatValue,
                    intValue: intValue
                )
            }

        case .not(let inner):
            return !inner.evaluate(
                stringValue: stringValue,
                floatValue: floatValue,
                intValue: intValue
            )
        }
    }
}
