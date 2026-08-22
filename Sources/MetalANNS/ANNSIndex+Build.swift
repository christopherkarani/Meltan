import Foundation
import Metal
import MetalANNSCore

extension GraphIndex {
    public func build(vectors inputVectors: [[Float]], ids: [String]) async throws {
        guard case .empty = lifecycle else {
            throw ANNSError.constructionFailed("Index is already built. Create a new GraphIndex to rebuild.")
        }
        guard !inputVectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build index with empty vectors")
        }
        guard inputVectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }

        let dim = inputVectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        for vector in inputVectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
        }
        guard inputVectors.count >= 2 else {
            throw ANNSError.constructionFailed("Build requires at least 2 vectors")
        }
        guard configuration.degree > 0 else {
            throw ANNSError.constructionFailed("Degree must be greater than zero")
        }
        guard configuration.degree < inputVectors.count else {
            throw ANNSError.constructionFailed(
                "Degree \(configuration.degree) must be less than node count \(inputVectors.count)"
            )
        }
        if configuration.metric == .hamming, !configuration.useBinary {
            throw ANNSError.constructionFailed("metric .hamming requires useBinary == true")
        }

        let capacity = max(2, inputVectors.count * 2)
        let device = context?.device
        let vectorBuffer: any VectorStorage
        if configuration.useBinary {
            guard configuration.metric == .hamming else {
                throw ANNSError.constructionFailed("useBinary requires metric == .hamming")
            }
            guard dim % 8 == 0 else {
                throw ANNSError.constructionFailed("Binary index requires dim % 8 == 0, got dim=\(dim)")
            }
            vectorBuffer = try BinaryVectorBuffer(capacity: capacity, dim: dim, device: device)
        } else if configuration.useFloat16 {
            vectorBuffer = try Float16VectorBuffer(capacity: capacity, dim: dim, device: device)
        } else {
            vectorBuffer = try VectorBuffer(capacity: capacity, dim: dim, device: device)
        }

        let graphBuffer = try GraphBuffer(capacity: capacity, degree: configuration.degree, device: device)
        try vectorBuffer.batchInsert(vectors: inputVectors, startingAt: 0)
        vectorBuffer.setCount(inputVectors.count)

        var builtIDMap = IDMap()
        for id in ids {
            guard builtIDMap.assign(externalID: id) != nil else {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        let builtEntryPoint: UInt32
        if let context, shouldUseGPUConstruction(nodeCount: inputVectors.count) {
            try await NNDescentGPU.build(
                context: context,
                vectors: vectorBuffer,
                graph: graphBuffer,
                nodeCount: inputVectors.count,
                metric: configuration.metric,
                maxIterations: configuration.maxIterations,
                convergenceThreshold: configuration.convergenceThreshold
            )
            builtEntryPoint = 0
        } else {
            let cpuVectors: [[Float]]
            if configuration.useBinary {
                cpuVectors = inputVectors.map(Self.quantizeForHamming)
            } else {
                cpuVectors = inputVectors
            }
            let cpuResult = try await NNDescentCPU.build(
                vectors: cpuVectors,
                degree: configuration.degree,
                metric: configuration.metric,
                maxIterations: configuration.maxIterations,
                convergenceThreshold: configuration.convergenceThreshold
            )

            for nodeID in 0..<cpuResult.graph.count {
                let neighbors = cpuResult.graph[nodeID]
                var neighborIDs = Array(repeating: UInt32.max, count: configuration.degree)
                var neighborDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
                for slot in 0..<min(configuration.degree, neighbors.count) {
                    neighborIDs[slot] = neighbors[slot].0
                    neighborDistances[slot] = neighbors[slot].1
                }
                try graphBuffer.setNeighbors(of: nodeID, ids: neighborIDs, distances: neighborDistances)
            }
            graphBuffer.setCount(inputVectors.count)
            builtEntryPoint = cpuResult.entryPoint
        }

        try GraphPruner.prune(
            graph: graphBuffer,
            vectors: vectorBuffer,
            nodeCount: inputVectors.count,
            metric: configuration.metric
        )

        self.idMap = builtIDMap
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.lifecycle = .ready(
            ReadyState(
                vectors: vectorBuffer,
                graph: graphBuffer,
                entryPoint: builtEntryPoint
            )
        )
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
    }

}
