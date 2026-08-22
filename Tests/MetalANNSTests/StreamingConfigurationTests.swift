import Foundation
import Testing

@testable import MetalANNS

@Suite("StreamingConfiguration Tests")
struct StreamingConfigurationTests {
    @Test("Default values")
    func defaultValues() {
        let configuration = StreamingConfiguration()

        #expect(configuration.deltaCapacity == 10_000)
        #expect(configuration.mergeStrategy == .background)
    }

    @Test("Custom values")
    func customValues() {
        let indexConfiguration = IndexConfiguration(
            degree: 64,
            metric: .l2,
            efConstruction: 200,
            efSearch: 128,
            maxIterations: 30,
            useFloat16: true,
            convergenceThreshold: 0.0005,
            hnswConfiguration: .default,
            repairConfiguration: .default
        )
        let configuration = StreamingConfiguration(
            deltaCapacity: 500,
            mergeStrategy: .blocking,
            indexConfiguration: indexConfiguration
        )

        #expect(configuration.deltaCapacity == 500)
        #expect(configuration.mergeStrategy == .blocking)
        #expect(configuration.indexConfiguration.degree == 64)
        #expect(configuration.indexConfiguration.metric.rawValue == Metric.l2.rawValue)
    }

    @Test("Codable round-trip")
    func codableRoundTrip() throws {
        let original = StreamingConfiguration(
            deltaCapacity: 500,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(
                degree: 48,
                metric: .innerProduct,
                efConstruction: 150,
                efSearch: 96,
                maxIterations: 25,
                useFloat16: true,
                convergenceThreshold: 0.002,
                hnswConfiguration: .default,
                repairConfiguration: .default
            )
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(StreamingConfiguration.self, from: data)

        #expect(decoded == original)
    }
}
