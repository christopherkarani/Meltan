import Foundation
import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "MetalDevice")

// Metal types (MTLDevice, MTLLibrary) are not marked Sendable by Apple,
// but this type is immutable after init and safe to share across isolation boundaries.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let queuePool: CommandQueuePool
    public let library: MTLLibrary
    public let pipelineCache: PipelineCache
    public let searchBufferPool: SearchBufferPool

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        guard let queue = device.makeCommandQueue() else {
            throw ANNSError.deviceNotSupported
        }

        let library = try Self.loadLibrary(for: device)

        self.device = device
        self.commandQueue = queue
        self.queuePool = try CommandQueuePool(
            device: device,
            count: max(1, min(ProcessInfo.processInfo.activeProcessorCount, 16))
        )
        self.library = library
        self.pipelineCache = PipelineCache(device: device, library: library)
        self.searchBufferPool = SearchBufferPool(device: device)

        logger.debug("MetalContext initialized: \(device.name)")
    }

    private static func loadLibrary(for device: MTLDevice) throws -> MTLLibrary {
        do {
            return try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            logger.warning("Falling back from Bundle.module default Metal library: \(error.localizedDescription)")
        }

        if let library = device.makeDefaultLibrary() {
            logger.debug("Loaded Metal shader library via device.makeDefaultLibrary() fallback")
            return library
        }

        do {
            let library = try makeLibraryFromBundledSources(device: device)
            logger.debug("Compiled Metal shader library from bundled source fallback")
            return library
        } catch {
            throw ANNSError.constructionFailed("Failed to load Metal shader library: \(error)")
        }
    }

    private static func makeLibraryFromBundledSources(device: MTLDevice) throws -> MTLLibrary {
        let bundle = Bundle.module
        let directURLs = bundle.urls(forResourcesWithExtension: "metal", subdirectory: "Shaders") ?? []
        let flattenedURLs = bundle.urls(forResourcesWithExtension: "metal", subdirectory: nil) ?? []
        let shaderURLs = (directURLs + flattenedURLs)
            .reduce(into: [URL]()) { partial, url in
                if !partial.contains(url) {
                    partial.append(url)
                }
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !shaderURLs.isEmpty else {
            throw ANNSError.constructionFailed("No bundled Metal shader sources were found")
        }

        let source =
            try shaderURLs
            .map { try String(contentsOf: $0, encoding: .utf8) }
            .joined(separator: "\n\n")
        return try device.makeLibrary(source: source, options: nil)
    }

    public func execute(_ encode: (MTLCommandBuffer) throws -> Void) async throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw ANNSError.gpuResourceExhausted("Failed to create command buffer")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw ANNSError.gpuExecutionFailed("Command buffer failed: \(error)")
        }
    }

    /// Like execute(), but picks a queue from the pool for pipelining concurrent calls.
    public func executeOnPool(_ encode: (MTLCommandBuffer) throws -> Void) async throws {
        let queue = await queuePool.next()
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ANNSError.gpuResourceExhausted("Failed to create command buffer from pool queue")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw ANNSError.gpuExecutionFailed("Command buffer failed: \(error.localizedDescription)")
        }
    }
}
