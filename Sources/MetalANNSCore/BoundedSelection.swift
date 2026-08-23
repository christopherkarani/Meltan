import Foundation

struct BinaryHeap<Element> {
    private var storage: [Element] = []
    private let areSorted: (Element, Element) -> Bool

    init(sort: @escaping (Element, Element) -> Bool) {
        self.areSorted = sort
    }

    var count: Int { storage.count }
    var isEmpty: Bool { storage.isEmpty }
    var peek: Element? { storage.first }

    mutating func push(_ element: Element) {
        storage.append(element)
        siftUp(from: storage.count - 1)
    }

    /// Preallocates backing storage so pushes up to this count never reallocate.
    mutating func reserveCapacity(_ capacity: Int) {
        storage.reserveCapacity(capacity)
    }

    @discardableResult
    mutating func pop() -> Element? {
        guard !storage.isEmpty else {
            return nil
        }
        if storage.count == 1 {
            return storage.removeLast()
        }

        let first = storage[0]
        storage[0] = storage.removeLast()
        siftDown(from: 0)
        return first
    }

    mutating func replaceTop(with element: Element) {
        guard !storage.isEmpty else {
            push(element)
            return
        }
        storage[0] = element
        siftDown(from: 0)
    }

    func unorderedElements() -> [Element] {
        storage
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2

        while child > 0 && areSorted(storage[child], storage[parent]) {
            storage.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index

        while true {
            let left = (2 * parent) + 1
            let right = left + 1
            var candidate = parent

            if left < storage.count && areSorted(storage[left], storage[candidate]) {
                candidate = left
            }
            if right < storage.count && areSorted(storage[right], storage[candidate]) {
                candidate = right
            }
            if candidate == parent {
                return
            }

            storage.swapAt(parent, candidate)
            parent = candidate
        }
    }
}

struct BoundedPriorityBuffer<Element> {
    private(set) var heap: BinaryHeap<Element>
    let capacity: Int
    private let outranks: (Element, Element) -> Bool

    init(capacity: Int, outranks: @escaping (Element, Element) -> Bool) {
        self.capacity = max(0, capacity)
        self.outranks = outranks
        // Keep the worst element at the root so replacement is O(log k).
        var heap = BinaryHeap(sort: { lhs, rhs in outranks(rhs, lhs) })
        heap.reserveCapacity(self.capacity)
        self.heap = heap
    }

    var count: Int { heap.count }
    var worst: Element? { heap.peek }

    mutating func insert(_ element: Element) {
        guard capacity > 0 else {
            return
        }
        if heap.count < capacity {
            heap.push(element)
            return
        }
        guard let worst else {
            heap.push(element)
            return
        }
        guard outranks(element, worst) else {
            return
        }
        heap.replaceTop(with: element)
    }

    func sortedElements() -> [Element] {
        heap.unorderedElements().sorted(by: outranks)
    }

    /// Best-first elements via in-place heapsort over the stored layout.
    /// Deterministic for equal-rank elements; matches legacy bounded-heap
    /// extraction order.
    func heapsortedElements() -> [Element] {
        var result = heap.unorderedElements()
        var count = result.count
        while count > 1 {
            result.swapAt(0, count - 1)
            count -= 1
            var position = 0
            while true {
                let left = 2 * position + 1
                let right = left + 1
                var worst = position
                if left < count, outranks(result[worst], result[left]) { worst = left }
                if right < count, outranks(result[worst], result[right]) { worst = right }
                if worst == position { break }
                result.swapAt(position, worst)
                position = worst
            }
        }
        return result
    }
}

/// Fixed-capacity list kept sorted best-first under `outranks`.
/// Insertion is O(log n) search + shift; equal-rank elements keep the
/// most recently inserted ahead of earlier ones.
package struct BoundedSortedList<Element> {
    let capacity: Int
    package private(set) var elements: [Element] = []
    private let outranks: @Sendable (Element, Element) -> Bool

    package init(capacity: Int, outranks: @escaping @Sendable (Element, Element) -> Bool) {
        self.capacity = max(0, capacity)
        self.outranks = outranks
        elements.reserveCapacity(self.capacity)
    }

    var count: Int { elements.count }

    package mutating func insert(_ element: Element) {
        guard capacity > 0 else {
            return
        }
        if let worst = elements.last, elements.count >= capacity, !outranks(element, worst) {
            return
        }

        let insertionIndex = lowerBound(of: element)
        elements.insert(element, at: insertionIndex)
        if elements.count > capacity {
            elements.removeLast()
        }
    }

    package mutating func insert(contentsOf newElements: [Element]) {
        for element in newElements {
            insert(element)
        }
    }

    private func lowerBound(of element: Element) -> Int {
        var low = 0
        var high = elements.count

        while low < high {
            let mid = (low + high) / 2
            if outranks(elements[mid], element) {
                low = mid + 1
            } else {
                high = mid
            }
        }

        return low
    }
}

extension BoundedSortedList: Sendable where Element: Sendable {}
