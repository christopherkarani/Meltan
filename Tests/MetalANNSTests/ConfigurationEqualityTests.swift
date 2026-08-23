import Testing

@testable import MetalANNS
@testable import MetalANNSCore

@Suite("ConfigurationEquality Tests")
struct ConfigurationEqualityTests {
    @Test("Identical configurations are equal")
    func identicalConfigurationsAreEqual() {
        #expect(IndexConfiguration() == IndexConfiguration())
        #expect(HNSWConfiguration() == HNSWConfiguration())
        #expect(RepairConfiguration() == RepairConfiguration())
        #expect(StreamingConfiguration() == StreamingConfiguration())
    }

    @Test("Index configurations differing only in exactSearchMaxVectorCount are unequal")
    func indexConfigurationsDifferingOnlyInExactSearchCeilingAreUnequal() {
        let lhs = IndexConfiguration(exactSearchMaxVectorCount: 10_000)
        let rhs = IndexConfiguration(exactSearchMaxVectorCount: 20_000)

        #expect(lhs != rhs)
        #expect(lhs.exactSearchMaxVectorCount != rhs.exactSearchMaxVectorCount)
    }

    @Test("Streaming configurations differing only in exactSearchMaxVectorCount are unequal")
    func streamingConfigurationsDifferingOnlyInExactSearchCeilingAreUnequal() {
        let lhs = StreamingConfiguration(
            deltaCapacity: 500,
            indexConfiguration: IndexConfiguration(exactSearchMaxVectorCount: 10_000)
        )
        let rhs = StreamingConfiguration(
            deltaCapacity: 500,
            indexConfiguration: IndexConfiguration(exactSearchMaxVectorCount: 20_000)
        )

        #expect(lhs != rhs)
    }

    @Test("HNSW configuration changes participate in equality")
    func hnswConfigurationChangesParticipateInEquality() {
        let lhs = IndexConfiguration(hnswConfiguration: .default)
        let rhs = IndexConfiguration(
            hnswConfiguration: HNSWConfiguration(
                enabled: true,
                M: 16,
                maxLayers: 6
            )
        )
        let streamingLhs = StreamingConfiguration(indexConfiguration: .default)
        let streamingRhs = StreamingConfiguration(indexConfiguration: rhs)

        #expect(lhs.hnswConfiguration.M != rhs.hnswConfiguration.M)
        #expect(lhs != rhs)
        #expect(streamingLhs != streamingRhs)
    }

    @Test("Repair configuration changes participate in equality")
    func repairConfigurationChangesParticipateInEquality() {
        let lhs = IndexConfiguration(repairConfiguration: .default)
        let rhs = IndexConfiguration(
            repairConfiguration: RepairConfiguration(
                repairInterval: 200,
                repairDepth: 2,
                repairIterations: 5,
                enabled: true
            )
        )
        let streamingLhs = StreamingConfiguration(indexConfiguration: .default)
        let streamingRhs = StreamingConfiguration(indexConfiguration: rhs)

        #expect(lhs.repairConfiguration.repairInterval != rhs.repairConfiguration.repairInterval)
        #expect(lhs != rhs)
        #expect(streamingLhs != streamingRhs)
    }
}
