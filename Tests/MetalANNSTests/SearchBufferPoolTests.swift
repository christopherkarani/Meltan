import Metal
import Testing

@testable import MetalANNSCore

@Suite("SearchBufferPool Tests")
struct SearchBufferPoolTests {
    @Test func acquireAndReleaseReturnsSameBuffer() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(device: device)

        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let ptr1 = b1.queryBuffer.gpuAddress
        pool.release(b1)

        let b2 = try pool.acquire(queryDim: 128, maxK: 10)
        #expect(b2.queryBuffer.gpuAddress == ptr1)
        pool.release(b2)
    }

    @Test func acquireLargerDimAllocatesNew() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(device: device)

        let small = try pool.acquire(queryDim: 64, maxK: 10)
        pool.release(small)

        let large = try pool.acquire(queryDim: 512, maxK: 10)
        #expect(large.queryBuffer.length >= 512 * MemoryLayout<Float>.stride)
        pool.release(large)
    }

    @Test func concurrentAcquireReturnsDistinctBuffers() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(device: device)

        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let b2 = try pool.acquire(queryDim: 128, maxK: 10)
        #expect(b1.queryBuffer.gpuAddress != b2.queryBuffer.gpuAddress)
        pool.release(b1)
        pool.release(b2)
    }

    @Test func releaseEvictsToEntryCap() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(
            device: device,
            maxRetainedEntries: 1,
            maxRetainedBytes: .max
        )

        let small = try pool.acquire(queryDim: 64, maxK: 10)
        pool.release(small)
        #expect(pool.availableCountForTesting == 1)

        let large = try pool.acquire(queryDim: 256, maxK: 10)
        let expectedRetainedBytes =
            large.queryBuffer.length
            + large.outputDistanceBuffer.length
            + large.outputIDBuffer.length
        pool.release(large)

        #expect(pool.availableCountForTesting == 1)
        #expect(pool.retainedBytesForTesting == expectedRetainedBytes)
    }

    @Test func releaseEvictsToByteCap() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(
            device: device,
            maxRetainedEntries: 8,
            maxRetainedBytes: 1_000
        )

        let first = try pool.acquire(queryDim: 100, maxK: 10)
        pool.release(first)
        #expect(pool.availableCountForTesting == 1)

        let second = try pool.acquire(queryDim: 200, maxK: 10)
        let secondBytes =
            second.queryBuffer.length
            + second.outputDistanceBuffer.length
            + second.outputIDBuffer.length
        pool.release(second)

        #expect(pool.availableCountForTesting == 1)
        #expect(pool.retainedBytesForTesting == secondBytes)
        #expect(pool.retainedBytesForTesting <= 1_000)
    }

    @Test func releaseDropsOversizedBuffer() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(
            device: device,
            maxRetainedEntries: 8,
            maxRetainedBytes: 1_024
        )

        let oversized = try pool.acquire(queryDim: 1_024, maxK: 10)
        pool.release(oversized)

        #expect(pool.availableCountForTesting == 0)
        #expect(pool.retainedBytesForTesting == 0)
    }

    @Test func acquireAndReleaseVisitedReturnsSameBuffer() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(device: device)

        let v1 = try pool.acquireVisited(nodeCount: 1000)
        let ptr1 = v1.buffer.gpuAddress
        pool.releaseVisited(v1.buffer, capacity: 1000)

        let v2 = try pool.acquireVisited(nodeCount: 1000)
        #expect(v2.buffer.gpuAddress == ptr1, "Pooled visited buffer should be reused")
        pool.releaseVisited(v2.buffer, capacity: 1000)
    }

    @Test func visitedGenerationsAreMonotonicallyIncreasing() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(device: device)

        let v1 = try pool.acquireVisited(nodeCount: 100)
        let gen1 = v1.generation
        pool.releaseVisited(v1.buffer, capacity: 100)

        let v2 = try pool.acquireVisited(nodeCount: 100)
        let gen2 = v2.generation
        pool.releaseVisited(v2.buffer, capacity: 100)

        #expect(gen2 > gen1, "Each acquire should return a strictly higher generation")
        #expect(gen1 != 0, "Generation must never be 0 (reserved as unvisited sentinel)")
        #expect(gen2 != 0, "Generation must never be 0 (reserved as unvisited sentinel)")
    }

    @Test func concurrentVisitedAcquireReturnsDistinctBuffers() throws {
        let device = try Require.metalDevice()
        let pool = SearchBufferPool(device: device)

        let v1 = try pool.acquireVisited(nodeCount: 100)
        let v2 = try pool.acquireVisited(nodeCount: 100)
        #expect(
            v1.buffer.gpuAddress != v2.buffer.gpuAddress,
            "Concurrent acquires must return distinct buffers"
        )
        pool.releaseVisited(v1.buffer, capacity: 100)
        pool.releaseVisited(v2.buffer, capacity: 100)
    }

    @Test func fullGPUSearchCorrectAfterPoolRefactor() async throws {
        let device = try Require.metalDevice()
        let context = try MetalContext()

        let dim = 16
        let nodeCount = 32
        let degree = 8
        var rng = SeededGenerator(state: 42)
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        for (i, vector) in vectors.enumerated() {
            try vectorBuffer.insert(vector: vector, at: i)
        }
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        try await NNDescentGPU.build(
            context: context,
            vectors: vectorBuffer,
            graph: graph,
            nodeCount: nodeCount,
            metric: .cosine
        )

        let r1 = try await FullGPUSearch.search(
            context: context,
            query: vectors[0],
            vectors: vectorBuffer,
            graph: graph,
            entryPoint: 0,
            k: 5,
            ef: 16,
            metric: .cosine
        )
        let r2 = try await FullGPUSearch.search(
            context: context,
            query: vectors[1],
            vectors: vectorBuffer,
            graph: graph,
            entryPoint: 0,
            k: 5,
            ef: 16,
            metric: .cosine
        )

        #expect(r1.count > 0, "First search returned no results")
        #expect(r2.count > 0, "Second search returned no results")
        #expect(r1[0].score < 0.1, "First result should be near-exact match (query is in index)")
    }
}
