import Accelerate
import Foundation
import Metal

extension ResidualCascade {
    struct GPUResidualBoundParameters {
        let vectorCount: UInt32
        let headWidth: UInt32
        let metricType: UInt32
        let queryTailNorm: Float
        let queryNorm: Float
        let queryNormSq: Float
        let dotMean: Float
        let meanNormSq: Float
        let slack: Float
        let absoluteSlack: Float
    }

    private struct GPUQueryInput {
        let context: MetalContext
        let query: [Float]
        let corpus: UnsafePointer<Float>
        let rowCount: Int
        let dimensionCount: Int
        let effectiveK: Int
        let metric: Metric
        let aux: BoundBuffer
        let boundContext: ResidualCascadeMath.QueryBoundContext
        let queryNormSq: Float
    }

    private struct GPUDispatchInput {
        let query: GPUQueryInput
        let planes: BoundBuffer.GPUPlanes
        let lowerBoundBuffer: MTLBuffer
        let pipeline: MTLComputePipelineState
        let parameters: GPUResidualBoundParameters
    }

    /// GPU-accelerated bound pass with the same exact host verification and
    /// selection phases as `search`. Returns nil if the GPU resource path is
    /// unavailable, allowing FlatGPUSearch to use its normal fallback.
    static func searchGPU(
        context: MetalContext,
        query: [Float],
        vectors: any VectorStorage,
        neighborTotal: Int,
        metric: Metric
    ) async -> [SearchResult]? {
        guard let vectorBuffer = vectors as? VectorBuffer,
            !vectors.isFloat16,
            let corpus = vectorBuffer.floatPointer.baseAddress
        else { return nil }
        let rowCount = vectors.count
        let dimensionCount = vectors.dim
        guard rowCount >= minVectorCount, dimensionCount > 0,
            query.count == dimensionCount, neighborTotal > 0, metric != .hamming
        else { return nil }

        let cacheKey = Key(
            bufferID: ObjectIdentifier(vectorBuffer.buffer),
            bufferLength: vectorBuffer.buffer.length,
            rowCount: rowCount,
            dimensionCount: dimensionCount
        )
        let aux: BoundBuffer
        if let cached = store.get(cacheKey), cached.rowCount >= rowCount {
            aux = cached
        } else {
            guard let built = build(
                corpus: corpus, rowCount: rowCount,
                dimensionCount: dimensionCount, key: cacheKey
            ) else { return nil }
            aux = built
        }

        let effectiveK = min(neighborTotal, FlatGPUSearch.maxTopK, rowCount)
        guard effectiveK > 0 else { return nil }
        var queryNormSq: Float = 0
        vDSP_dotpr(query, 1, query, 1, &queryNormSq, vDSP_Length(dimensionCount))
        if metric == .cosine && queryNormSq < 1e-20 {
            return lowestIdsResult(count: rowCount, take: effectiveK)
        }

        let boundContext = ResidualCascadeMath.prepareQueryContext(
            query: query, aux: aux, dimensionCount: dimensionCount
        )
        return await runGPUQuery(GPUQueryInput(
            context: context, query: query, corpus: corpus,
            rowCount: rowCount, dimensionCount: dimensionCount,
            effectiveK: effectiveK, metric: metric, aux: aux,
            boundContext: boundContext, queryNormSq: queryNormSq
        ))
    }

    private static func runGPUQuery(_ input: GPUQueryInput) async -> [SearchResult]? {
        guard let planes = input.aux.gpuPlanes(device: input.context.device),
            let lowerBoundBuffer = input.context.device.makeBuffer(
                length: max(input.rowCount * MemoryLayout<Float>.stride, 4),
                options: .storageModeShared
            ) else { return nil }

        let metricType: UInt32 = switch input.metric {
        case .cosine: 0
        case .l2: 1
        case .innerProduct: 2
        case .hamming: 3
        }
        let parameters = GPUResidualBoundParameters(
            vectorCount: UInt32(input.rowCount),
            headWidth: UInt32(input.aux.headWidth),
            metricType: metricType,
            queryTailNorm: input.boundContext.tailNorm,
            queryNorm: input.boundContext.norm,
            queryNormSq: input.boundContext.normSq,
            dotMean: input.boundContext.dotMean,
            meanNormSq: input.aux.meanNormSq,
            slack: boundSlack,
            absoluteSlack: boundAbsSlack
        )

        do {
            let pipeline = try await input.context.pipelineCache.pipeline(
                for: "residual_compute_bounds"
            )
            let phaseStart = DispatchTime.now()
            try await dispatchGPUBounds(GPUDispatchInput(
                query: input,
                planes: planes,
                lowerBoundBuffer: lowerBoundBuffer,
                pipeline: pipeline,
                parameters: parameters
            ))
            if ProcessInfo.processInfo.environment["METALANNS_RESIDUAL_STATS"] == "1" {
                let elapsed = Double(
                    DispatchTime.now().uptimeNanoseconds - phaseStart.uptimeNanoseconds
                ) / 1_000_000
                FileHandle.standardError.write(Data(
                    String(format: "[ResidualCascade] gpu_p1_bounds=%.2fms\n", elapsed).utf8
                ))
            }
        } catch {
            return nil
        }

        let lowerBounds = Array(UnsafeBufferPointer(
            start: lowerBoundBuffer.contents().assumingMemoryBound(to: Float.self),
            count: input.rowCount
        ))
        let statsEnabled = ProcessInfo.processInfo.environment["METALANNS_RESIDUAL_STATS"] == "1"
        var phaseTimings: [(String, Double)] = []
        func recordPhase(_ name: String, _ start: DispatchTime) {
            if statsEnabled {
                phaseTimings.append((name, Double(
                    DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
                ) / 1_000_000))
            }
        }
        let results = resolveTopK(ResolveInput(
            lowerBounds: lowerBounds,
            corpus: input.corpus,
            dimensionCount: input.dimensionCount,
            query: input.query,
            queryNormSq: input.queryNormSq,
            aux: input.aux,
            metric: input.metric,
            effectiveK: input.effectiveK,
            rowCount: input.rowCount,
            recordPhase: recordPhase,
            statsEnabled: statsEnabled
        ))
        if statsEnabled {
            let summary = phaseTimings.map {
                String(format: "%@=%.2fms", $0.0, $0.1)
            }.joined(separator: " ")
            FileHandle.standardError.write(Data(
                "[ResidualCascade] gpu_phases: \(summary)\n".utf8
            ))
        }
        return results
    }

    private static func dispatchGPUBounds(_ input: GPUDispatchInput) async throws {
        try await input.query.context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.searchFailed("Failed to create residual bound encoder")
            }
            defer { encoder.endEncoding() }
            encoder.setComputePipelineState(input.pipeline)
            encoder.setBuffer(input.planes.projection, offset: 0, index: 0)
            encoder.setBuffer(input.planes.vDotMu, offset: 0, index: 1)
            encoder.setBuffer(input.planes.rowNormSq, offset: 0, index: 2)
            encoder.setBuffer(input.planes.tailNorm, offset: 0, index: 3)
            input.query.boundContext.headDots.withUnsafeBufferPointer { queryBuffer in
                encoder.setBytes(
                    queryBuffer.baseAddress!,
                    length: queryBuffer.count * MemoryLayout<Float>.stride,
                    index: 4
                )
            }
            encoder.setBuffer(input.lowerBoundBuffer, offset: 0, index: 5)
            var parameters = input.parameters
            withUnsafePointer(to: &parameters) { parameterPointer in
                encoder.setBytes(
                    parameterPointer,
                    length: MemoryLayout<GPUResidualBoundParameters>.stride,
                    index: 6
                )
            }
            let threads = MTLSize(width: input.query.rowCount, height: 1, depth: 1)
            var groupWidth = min(256, input.pipeline.maxTotalThreadsPerThreadgroup)
            groupWidth -= groupWidth % 32
            groupWidth = max(groupWidth, 32)
            encoder.dispatchThreads(
                threads,
                threadsPerThreadgroup: MTLSize(
                    width: groupWidth, height: 1, depth: 1
                )
            )
        }
    }
}
