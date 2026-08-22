import Testing

@testable import MetalANNSCore

@Suite("MetalContext Multi-Queue Tests")
struct MetalContextMultiQueueTests {
    @Test("poolInitialisedOnContext")
    func poolInitialisedOnContext() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            let context = try Require.metalContext()
            #expect(await context.queuePool.queues.isEmpty == false)
        #endif
    }

    @Test("executeOnPoolUsesPoolQueue")
    func executeOnPoolUsesPoolQueue() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            let context = try Require.metalContext()

            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<2 {
                    group.addTask {
                        try await context.executeOnPool { _ in }
                    }
                }

                for try await _ in group {}
            }
        #endif
    }

    @Test("legacyExecuteUnchanged")
    func legacyExecuteUnchanged() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            let context = try Require.metalContext()
            try await context.execute { _ in }
        #endif
    }
}
