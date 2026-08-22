import Metal
import Testing

@testable import MetalANNSCore

@Suite("CommandQueuePool Tests")
struct CommandQueuePoolTests {
    @Test("createsNQueues")
    func createsNQueues() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            guard let device = MTLCreateSystemDefaultDevice() else {
                return
            }

            let pool = try CommandQueuePool(device: device, count: 4)
            let queues = await pool.queues
            #expect(queues.count == 4)
        #endif
    }

    @Test("queuesAreDistinct")
    func queuesAreDistinct() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            guard let device = MTLCreateSystemDefaultDevice() else {
                return
            }

            let pool = try CommandQueuePool(device: device, count: 4)
            let ids = await Set(pool.queues.map(ObjectIdentifier.init))
            #expect(ids.count == 4)
        #endif
    }

    @Test("nextIsRoundRobin")
    func nextIsRoundRobin() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            guard let device = MTLCreateSystemDefaultDevice() else {
                return
            }

            let pool = try CommandQueuePool(device: device, count: 4)
            var ids: [ObjectIdentifier] = []
            ids.reserveCapacity(8)

            for _ in 0..<8 {
                let queue = await pool.next()
                ids.append(ObjectIdentifier(queue))
            }

            #expect(Array(ids[0..<4]) == Array(ids[4..<8]))
        #endif
    }

    @Test("concurrentNextIsSafe")
    func concurrentNextIsSafe() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            guard let device = MTLCreateSystemDefaultDevice() else {
                return
            }

            let pool = try CommandQueuePool(device: device, count: 4)
            let picked = try await withThrowingTaskGroup(of: ObjectIdentifier.self) { group in
                for _ in 0..<8 {
                    group.addTask {
                        let queue = await pool.next()
                        return ObjectIdentifier(queue)
                    }
                }

                var ids: [ObjectIdentifier] = []
                for try await id in group {
                    ids.append(id)
                }
                return ids
            }

            #expect(picked.count == 8)
            #expect(Set(picked).count <= 4)
        #endif
    }
}
