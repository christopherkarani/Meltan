import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "PipelineCache")

package actor PipelineCache {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var cache: [String: MTLComputePipelineState] = [:]

    package init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    package func pipeline(for functionName: String) throws -> MTLComputePipelineState {
        if let cached = cache[functionName] {
            return cached
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw ANNSError.gpuPipelineUnavailable("Metal function '\(functionName)' not found")
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        cache[functionName] = pipeline
        logger.debug("Compiled pipeline: \(functionName)")
        return pipeline
    }
}
