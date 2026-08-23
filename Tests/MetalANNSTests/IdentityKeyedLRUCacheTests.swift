import Foundation
import Testing

@testable import MetalANNSCore

/// Regression suite pinning the shared identity-keyed LRU semantics both
/// derived-data caches rely on: fixed-capacity oldest-first eviction without
/// recency refresh on lookup, replace-in-place publication, and buffer-scoped
/// invalidation so a recycled MTLBuffer address can never serve stale data.
@Suite("Identity Keyed LRU Cache Tests")
struct IdentityKeyedLRUCacheTests {
    private final class Box {
        let tag: Int
        init(tag: Int) { self.tag = tag }
    }

    private struct TestKey: BufferIdentityKeyed {
        let bufferID: ObjectIdentifier
        let shape: Int
    }

    private static func key(_ id: AnyObject, _ shape: Int) -> TestKey {
        TestKey(bufferID: ObjectIdentifier(id), shape: shape)
    }

    @Test("Miss returns nil; hit returns the stored box")
    func getReturnsStoredValue() {
        let cache = IdentityKeyedLRUCache<TestKey, Box>(capacity: 2)
        let owner = NSObject()
        #expect(cache.get(Self.key(owner, 1)) == nil)

        let box = Box(tag: 7)
        cache.store(box, for: Self.key(owner, 1))
        #expect(cache.get(Self.key(owner, 1))?.tag == 7)
    }

    @Test("Eviction drops the oldest insertion and lookups do not refresh recency")
    func evictsOldestWithoutRecencyRefresh() {
        let cache = IdentityKeyedLRUCache<TestKey, Box>(capacity: 2)
        let owner = NSObject()
        let first = Box(tag: 1)
        let second = Box(tag: 2)
        let third = Box(tag: 3)

        cache.store(first, for: Self.key(owner, 1))
        cache.store(second, for: Self.key(owner, 2))

        _ = cache.get(Self.key(owner, 1))
        cache.store(third, for: Self.key(owner, 3))

        #expect(cache.get(Self.key(owner, 1)) == nil)
        #expect(cache.get(Self.key(owner, 2))?.tag == 2)
        #expect(cache.get(Self.key(owner, 3))?.tag == 3)
    }

    @Test("Storing over an existing key replaces it in place without eviction")
    func storeOverExistingReplacesInPlace() {
        let cache = IdentityKeyedLRUCache<TestKey, Box>(capacity: 1)
        let owner = NSObject()
        let original = Box(tag: 1)
        let replacement = Box(tag: 2)

        cache.store(original, for: Self.key(owner, 1))
        cache.store(replacement, for: Self.key(owner, 1))

        #expect(cache.get(Self.key(owner, 1))?.tag == 2)
    }

    @Test("Invalidation drops every entry of one buffer and keeps the rest")
    func invalidateScopesToSingleBuffer() {
        let cache = IdentityKeyedLRUCache<TestKey, Box>(capacity: 8)
        let doomed = NSObject()
        let survivor = NSObject()

        cache.store(Box(tag: 1), for: Self.key(doomed, 1))
        cache.store(Box(tag: 2), for: Self.key(doomed, 2))
        cache.store(Box(tag: 3), for: Self.key(survivor, 1))

        cache.invalidate(bufferID: ObjectIdentifier(doomed))

        #expect(cache.get(Self.key(doomed, 1)) == nil)
        #expect(cache.get(Self.key(doomed, 2)) == nil)
        #expect(cache.get(Self.key(survivor, 1))?.tag == 3)
    }

    @Test("Clear empties every entry")
    func clearRemovesEverything() {
        let cache = IdentityKeyedLRUCache<TestKey, Box>(capacity: 4)
        let owner = NSObject()

        cache.store(Box(tag: 1), for: Self.key(owner, 1))
        cache.store(Box(tag: 2), for: Self.key(owner, 2))
        cache.clear()

        #expect(cache.get(Self.key(owner, 1)) == nil)
        #expect(cache.get(Self.key(owner, 2)) == nil)
    }
}
