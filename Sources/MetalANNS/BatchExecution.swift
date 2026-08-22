import Foundation

/// Ordered, bounded-concurrency batch fan-out over a throwing async operation.
///
/// Contract:
/// - Output element `i` is the result of applying `operation` to input element
///   `i`, regardless of completion order.
/// - At most `maxConcurrency` child tasks are in flight; each completion
///   refills the window until every input has run exactly once.
/// - Fail-fast: the first child error propagates unchanged to the caller and
///   cancels all remaining children. No partial results are ever returned —
///   either every element succeeded, or the call throws.
/// - Empty input yields an empty output without spawning any task.
///
/// Completeness needs no force-unwrap: one child runs per input, each reports
/// its own position exactly once, and the consuming loop drains the group
/// before returning — so sorting the collected `(position, output)` pairs by
/// position restores the input order with every slot guaranteed present.
enum BatchExecution {
    static func run<Element: Sendable, Output: Sendable>(
        over elements: [Element],
        maxConcurrency: Int,
        _ operation: @escaping @Sendable (Element) async throws -> Output
    ) async throws -> [Output] {
        guard !elements.isEmpty else {
            return []
        }
        let windowSize = max(1, min(maxConcurrency, elements.count))

        return try await withThrowingTaskGroup(
            of: (index: Int, output: Output).self
        ) { group in
            var nextIndex = 0

            for _ in 0..<windowSize {
                let index = nextIndex
                let element = elements[index]
                nextIndex += 1
                group.addTask {
                    let output = try await operation(element)
                    return (index: index, output: output)
                }
            }

            var collected: [(index: Int, output: Output)] = []
            collected.reserveCapacity(elements.count)

            for try await pair in group {
                collected.append(pair)

                if nextIndex < elements.count {
                    let index = nextIndex
                    let element = elements[index]
                    nextIndex += 1
                    group.addTask {
                        let output = try await operation(element)
                        return (index: index, output: output)
                    }
                }
            }

            return collected.sorted { $0.index < $1.index }.map(\.output)
        }
    }
}
