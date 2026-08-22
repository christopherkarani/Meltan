import Metal
import Testing

@testable import MetalANNSCore

@Suite("MetalDevice Tests")
struct MetalDeviceTests {
    @Test("MetalContext initializes on device with GPU")
    func initContext() throws {
        #if targetEnvironment(simulator)
            return
        #else
            let context = try Require.metalContext()
            #expect(context.device.name.isEmpty == false)
        #endif
    }

    @Test("PipelineCache compiles a function from the shader library")
    func pipelineCacheCompile() async throws {
        #if targetEnvironment(simulator)
            return
        #else
            let context = try Require.metalContext()
            let pipeline = try await context.pipelineCache.pipeline(for: "cosine_distance")
            #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        #endif
    }
}
