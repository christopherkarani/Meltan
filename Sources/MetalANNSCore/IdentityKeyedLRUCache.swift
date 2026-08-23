import Foundation

/// Key whose entries carry the identity of the MTLBuffer their derived data
/// was computed over, so invalidation can drop every entry for one buffer and
/// a recycled buffer address can never be served a predecessor's data.
package protocol BufferIdentityKeyed: Hashable {
    /// Stable identity of the backing MTLBuffer (`ObjectIdentifier(buffer)`).
    var bufferID: ObjectIdentifier { get }
}

/// Single generic identity-keyed cache owning the lock/get/store/invalidate
/// skeleton previously duplicated across the derived-data caches.
///
/// Fixed capacity; beyond it the oldest-inserted entry is evicted first
/// (lookups do not refresh recency).
///
/// Thread-safety: every operation synchronizes via an internal NSLock that
/// guards dictionary access only — never downstream computation such as norm
/// scans or quantization fills, which would otherwise serialize every
/// concurrent search. Values are therefore immutable snapshot boxes handed
/// out by reference: publication replaces the dictionary reference atomically
/// and in-flight readers retain their snapshot.
package final class IdentityKeyedLRUCache<Key: BufferIdentityKeyed, Value>:
    @unchecked Sendable
{
    private let lock = NSLock()
    private var entries: [Key: Value] = [:]
    private var order: [Key] = []
    private let capacity: Int

    package init(capacity: Int) {
        self.capacity = capacity
    }

    package func get(_ key: Key) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        return entries[key]
    }

    package func store(_ value: Value, for key: Key) {
        lock.lock()
        defer { lock.unlock() }
        if entries[key] == nil {
            order.append(key)
            while order.count > capacity {
                entries[order.removeFirst()] = nil
            }
        }
        entries[key] = value
    }

    /// Drops every entry keyed to one buffer; mutable storage calls this on
    /// in-place writes and deinit so stale data can never be served again.
    package func invalidate(bufferID: ObjectIdentifier) {
        lock.lock()
        defer { lock.unlock() }
        let doomed = order.filter { $0.bufferID == bufferID }
        for key in doomed {
            entries[key] = nil
        }
        order.removeAll { $0.bufferID == bufferID }
    }

    package func clear() {
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll()
        order.removeAll()
    }
}
