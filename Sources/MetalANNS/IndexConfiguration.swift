import MetalANNSCore

/// Search backend selection. `.exact` is the default fused scan (recall@k = 1).
/// `.fast` opts into host IVF-flat (probed inverted lists, exact within the
/// probe). Default search must stay exact; `.fast` is never implied.
public enum SearchMode: String, Sendable, Codable, Equatable {
    case exact
    case fast
}

public struct IndexConfiguration: Sendable, Codable {
    /// Default ceiling (1M vectors) for the fused exact-search path.
    public static let defaultExactSearchMaxVectorCount = 1_000_000

    /// Graph out-degree used by NN-Descent and search.
    /// For GPU construction (`NNDescentGPU` + bitonic sort), this must be a power of two and `<= 64`.
    public var degree: Int
    public var metric: Metric
    public var efConstruction: Int
    public var efSearch: Int
    public var maxIterations: Int
    public var useFloat16: Bool
    public var useBinary: Bool
    public var convergenceThreshold: Float
    public var hnswConfiguration: HNSWConfiguration
    public var repairConfiguration: RepairConfiguration
    /// Vector-count ceiling for the fused exact-search GPU path (single dispatch
    /// brute force). `0` disables the path entirely. Above this limit search
    /// falls back to the graph traversal backend.
    public var exactSearchMaxVectorCount: Int
    /// `.fast` selects host IVF-flat (approximate). Default is `.exact`.
    public var searchMode: SearchMode
    /// Coarse k-means list count for `.fast`. Clamped at search time.
    public var ivfListCount: Int
    /// Number of inverted lists probed for `.fast`.
    public var ivfNProbe: Int

    public static let `default` = IndexConfiguration(
        degree: 32,
        metric: .cosine,
        efConstruction: 100,
        efSearch: 64,
        maxIterations: 20,
        useFloat16: false,
        useBinary: false,
        convergenceThreshold: 0.001,
        hnswConfiguration: .default,
        repairConfiguration: .default
    )

    public init(
        degree: Int = 32,
        metric: Metric = .cosine,
        efConstruction: Int = 100,
        efSearch: Int = 64,
        maxIterations: Int = 20,
        useFloat16: Bool = false,
        useBinary: Bool = false,
        convergenceThreshold: Float = 0.001,
        hnswConfiguration: HNSWConfiguration = .default,
        repairConfiguration: RepairConfiguration = .default,
        exactSearchMaxVectorCount: Int = IndexConfiguration.defaultExactSearchMaxVectorCount,
        searchMode: SearchMode = .exact,
        ivfListCount: Int = IVFFlatSearch.defaultListCount,
        ivfNProbe: Int = IVFFlatSearch.defaultNProbe
    ) {
        self.degree = degree
        self.metric = metric
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.maxIterations = maxIterations
        self.useFloat16 = useFloat16
        self.useBinary = useBinary
        self.convergenceThreshold = convergenceThreshold
        self.hnswConfiguration = hnswConfiguration
        self.repairConfiguration = repairConfiguration
        self.exactSearchMaxVectorCount = exactSearchMaxVectorCount
        self.searchMode = searchMode
        self.ivfListCount = max(1, ivfListCount)
        self.ivfNProbe = max(1, ivfNProbe)
    }

    /// Returns true when `degree` satisfies GPU NN-Descent kernel constraints.
    public var isDegreeCompatibleWithGPUConstruction: Bool {
        degree > 0
            && degree <= 64
            && (degree & (degree - 1)) == 0
    }

    /// Validates degree requirements for GPU NN-Descent construction.
    ///
    /// Runtime kernels still perform defensive checks; this surfaces failures at
    /// configuration/build validation time with a clearer message.
    public func validateGPUConstructionConstraints() throws {
        guard degree <= 64 else {
            throw ANNSError.constructionFailed("GPU NN-Descent requires degree <= 64, got \(degree)")
        }
        guard (degree & (degree - 1)) == 0 else {
            throw ANNSError.constructionFailed(
                "GPU NN-Descent requires degree to be a power of two for bitonic sort, got \(degree)"
            )
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        degree = try container.decode(Int.self, forKey: .degree)
        metric = try container.decode(Metric.self, forKey: .metric)
        efConstruction = try container.decode(Int.self, forKey: .efConstruction)
        efSearch = try container.decode(Int.self, forKey: .efSearch)
        maxIterations = try container.decode(Int.self, forKey: .maxIterations)
        useFloat16 = try container.decode(Bool.self, forKey: .useFloat16)
        useBinary = try container.decodeIfPresent(Bool.self, forKey: .useBinary) ?? false
        convergenceThreshold = try container.decode(Float.self, forKey: .convergenceThreshold)
        hnswConfiguration =
            try container.decodeIfPresent(
                HNSWConfiguration.self,
                forKey: .hnswConfiguration
            ) ?? .default
        repairConfiguration =
            try container.decodeIfPresent(
                RepairConfiguration.self,
                forKey: .repairConfiguration
            ) ?? .default
        exactSearchMaxVectorCount =
            try container.decodeIfPresent(
                Int.self,
                forKey: .exactSearchMaxVectorCount
            ) ?? IndexConfiguration.defaultExactSearchMaxVectorCount
        searchMode = try container.decodeIfPresent(SearchMode.self, forKey: .searchMode) ?? .exact
        ivfListCount =
            try container.decodeIfPresent(Int.self, forKey: .ivfListCount)
            ?? IVFFlatSearch.defaultListCount
        ivfNProbe =
            try container.decodeIfPresent(Int.self, forKey: .ivfNProbe)
            ?? IVFFlatSearch.defaultNProbe
    }
}
