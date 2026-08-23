import Foundation

/// Single generic workspace-buffer pool owning the acquire/release/trim skeleton
/// previously duplicated across the GPU engines' scratch-buffer pools.
///
/// Thread-safety: All mutable operations and reads are synchronized via an internal
/// NSLock, making instances safe for concurrent acquire/release across isolation
/// domains. Entries must be immutable after init and hold only MTLBuffer references
/// plus inert metadata, which is why pools and their entry types carry
/// @unchecked Sendable. `make` runs outside the lock, so lock holds never span
/// buffer allocations or command-buffer waits.
package final class WorkspaceBufferPool<Entry>: @unchecked Sendable {
    private let lock = NSLock()
    private var available: [Entry] = []
    private var retainedBytes: Int = 0
    private let maxRetainedEntries: Int
    private let maxRetainedBytes: Int
    private let entryBytes: (Entry) -> Int

    /// - Parameters:
    ///   - maxRetainedEntries: Maximum stored entries; excess is trimmed oldest-first.
    ///   - maxRetainedBytes: Upper bound on retained bytes; releases beyond it are
    ///     dropped instead of pooled.
    ///   - entryBytes: Retained-byte cost of an entry, driving the byte budget.
    package init(
        maxRetainedEntries: Int,
        maxRetainedBytes: Int = .max,
        entryBytes: @escaping (Entry) -> Int
    ) {
        self.maxRetainedEntries = max(0, maxRetainedEntries)
        self.maxRetainedBytes = max(0, maxRetainedBytes)
        self.entryBytes = entryBytes
    }

    /// Returns the first pooled entry accepted by `fits`, otherwise builds a fresh
    /// entry via `make` outside the lock.
    package func acquire(
        where fits: (Entry) -> Bool,
        make: () throws -> Entry
    ) rethrows -> Entry {
        lock.lock()
        if let index = available.firstIndex(where: fits) {
            let entry = available.remove(at: index)
            retainedBytes -= entryBytes(entry)
            lock.unlock()
            return entry
        }
        lock.unlock()

        return try make()
    }

    /// Returns an entry to the pool for future reuse.
    package func release(_ entry: Entry) {
        lock.lock()
        defer { lock.unlock() }

        let bytes = entryBytes(entry)
        guard maxRetainedEntries > 0, bytes <= maxRetainedBytes else {
            return
        }

        available.append(entry)
        retainedBytes += bytes
        trimIfNeeded()
    }

    var availableCountForTesting: Int {
        lock.lock()
        defer { lock.unlock() }
        return available.count
    }

    var retainedBytesForTesting: Int {
        lock.lock()
        defer { lock.unlock() }
        return retainedBytes
    }

    private func trimIfNeeded() {
        while available.count > maxRetainedEntries || retainedBytes > maxRetainedBytes {
            let removed = available.removeFirst()
            retainedBytes -= entryBytes(removed)
        }
    }
}
