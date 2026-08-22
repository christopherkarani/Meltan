import Testing

@testable import MetalANNS

@Suite("IndexConfiguration Tests")
struct ConfigurationTests {
    @Test("Default configuration has expected values")
    func defaultConfiguration() {
        let config = IndexConfiguration.default
        #expect(config.degree == 32)
        #expect(config.metric == .cosine)
        #expect(config.efConstruction == 100)
        #expect(config.efSearch == 64)
        #expect(config.maxIterations == 20)
        #expect(config.useFloat16 == false)
        #expect(config.hnswConfiguration.enabled == true)
        #expect(config.hnswConfiguration.M == 8)
        #expect(config.hnswConfiguration.maxLayers == 6)
    }

    @Test("Metric enum has all cases")
    func metricCases() {
        let metrics: [Metric] = [.cosine, .l2, .innerProduct, .hamming]
        #expect(metrics.count == 4)
    }

    @Test("ANNSError cases are distinct")
    func errorCases() {
        let error1 = ANNSError.deviceNotSupported
        let error2 = ANNSError.dimensionMismatch(expected: 128, got: 64)
        #expect(error1.localizedDescription.isEmpty == false)
        _ = error2
    }

    @Test("GPU construction degree constraints are surfaced in configuration")
    func gpuDegreeConstraints() throws {
        let valid = IndexConfiguration(degree: 32)
        #expect(valid.isDegreeCompatibleWithGPUConstruction == true)
        try valid.validateGPUConstructionConstraints()

        let notPowerOfTwo = IndexConfiguration(degree: 48)
        #expect(notPowerOfTwo.isDegreeCompatibleWithGPUConstruction == false)
        do {
            try notPowerOfTwo.validateGPUConstructionConstraints()
            #expect(Bool(false), "Expected validation error for non-power-of-two degree")
        } catch let error as ANNSError {
            guard case .constructionFailed = error else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
                return
            }
        }

        let tooLarge = IndexConfiguration(degree: 128)
        #expect(tooLarge.isDegreeCompatibleWithGPUConstruction == false)
    }
}
