import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("ComputeBackend Protocol Tests")
struct BackendProtocolTests {
    @Test("BackendFactory creates a backend without crashing")
    func backendCreation() async throws {
        let backend = try BackendFactory.makeBackend()
        #expect(backend != nil)
    }
}
