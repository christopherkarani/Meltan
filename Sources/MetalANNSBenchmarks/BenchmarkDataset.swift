import Foundation
import MetalANNSCore

public enum BenchmarkDatasetError: Error, Sendable, CustomStringConvertible {
    case invalidDataset(String)
    case invalidMagic([UInt8])
    case unsupportedVersion(UInt32)
    case invalidMetric(UInt32)
    case truncatedFile(expectedAtLeast: Int, actual: Int)

    public var description: String {
        switch self {
        case .invalidDataset(let message):
            return "Invalid dataset: \(message)"
        case .invalidMagic(let bytes):
            return "Invalid .annbin magic: \(bytes)"
        case .unsupportedVersion(let version):
            return "Unsupported .annbin version: \(version)"
        case .invalidMetric(let raw):
            return "Invalid metric raw value in .annbin: \(raw)"
        case .truncatedFile(let expectedAtLeast, let actual):
            return "Truncated .annbin file: expected at least \(expectedAtLeast) bytes, got \(actual)"
        }
    }
}

public struct BenchmarkDataset: Sendable {
    public let trainVectors: [[Float]]
    public let testVectors: [[Float]]
    public let groundTruth: [[UInt32]]
    public let dimension: Int
    public let metric: Metric
    public let neighborsCount: Int

    public init(
        trainVectors: [[Float]],
        testVectors: [[Float]],
        groundTruth: [[UInt32]],
        dimension: Int,
        metric: Metric,
        neighborsCount: Int
    ) {
        self.trainVectors = trainVectors
        self.testVectors = testVectors
        self.groundTruth = groundTruth
        self.dimension = dimension
        self.metric = metric
        self.neighborsCount = neighborsCount
    }

    public static func synthetic(
        trainCount: Int,
        testCount: Int,
        dimension: Int,
        k: Int = 100,
        metric: Metric = .cosine,
        seed: Int = 42
    ) -> BenchmarkDataset {
        let safeTrainCount = max(1, trainCount)
        let safeTestCount = max(1, testCount)
        let safeDimension = max(1, dimension)
        let safeK = max(1, min(k, safeTrainCount))

        let trainVectors = makeVectors(count: safeTrainCount, dim: safeDimension, seedOffset: seed)
        let testVectors = makeVectors(count: safeTestCount, dim: safeDimension, seedOffset: seed + 1_000_000)

        var groundTruth: [[UInt32]] = []
        groundTruth.reserveCapacity(safeTestCount)
        for query in testVectors {
            groundTruth.append(bruteForceTopK(query: query, vectors: trainVectors, k: safeK, metric: metric))
        }

        return BenchmarkDataset(
            trainVectors: trainVectors,
            testVectors: testVectors,
            groundTruth: groundTruth,
            dimension: safeDimension,
            metric: metric,
            neighborsCount: safeK
        )
    }

    public func save(to path: String) throws {
        try validate()

        var data = Data()
        data.append(contentsOf: [0x41, 0x4E, 0x4E, 0x42])  // "ANNB"
        Self.appendUInt32(1, to: &data)
        Self.appendUInt32(UInt32(trainVectors.count), to: &data)
        Self.appendUInt32(UInt32(testVectors.count), to: &data)
        Self.appendUInt32(UInt32(dimension), to: &data)
        Self.appendUInt32(UInt32(neighborsCount), to: &data)
        Self.appendUInt32(Self.metricRaw(metric), to: &data)
        Self.appendUInt32(0, to: &data)
        Self.appendUInt32(0, to: &data)
        Self.appendUInt32(0, to: &data)

        for vector in trainVectors {
            for value in vector {
                Self.appendFloat32(value, to: &data)
            }
        }

        for vector in testVectors {
            for value in vector {
                Self.appendFloat32(value, to: &data)
            }
        }

        for neighbors in groundTruth {
            for id in neighbors {
                Self.appendUInt32(id, to: &data)
            }
        }

        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
    }

    public static func load(from path: String) throws -> BenchmarkDataset {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)

        if data.count < 40 {
            throw BenchmarkDatasetError.truncatedFile(expectedAtLeast: 40, actual: data.count)
        }

        let magic = Array(data[0..<4])
        guard magic == [0x41, 0x4E, 0x4E, 0x42] else {
            throw BenchmarkDatasetError.invalidMagic(magic)
        }

        var cursor = 4
        let version = try readUInt32(from: data, cursor: &cursor)
        guard version == 1 else {
            throw BenchmarkDatasetError.unsupportedVersion(version)
        }

        let trainCount = Int(try readUInt32(from: data, cursor: &cursor))
        let testCount = Int(try readUInt32(from: data, cursor: &cursor))
        let dimension = Int(try readUInt32(from: data, cursor: &cursor))
        let neighborsCount = Int(try readUInt32(from: data, cursor: &cursor))
        let metricRawValue = try readUInt32(from: data, cursor: &cursor)
        _ = try readUInt32(from: data, cursor: &cursor)
        _ = try readUInt32(from: data, cursor: &cursor)
        _ = try readUInt32(from: data, cursor: &cursor)

        guard dimension > 0 else {
            throw BenchmarkDatasetError.invalidDataset("dimension must be > 0")
        }

        guard neighborsCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("neighborsCount must be > 0")
        }

        guard let metric = metric(fromRaw: metricRawValue) else {
            throw BenchmarkDatasetError.invalidMetric(metricRawValue)
        }

        let expectedTrainValues = trainCount * dimension
        let expectedTestValues = testCount * dimension
        let expectedGroundTruthValues = testCount * neighborsCount
        let expectedBytes = 40 + (expectedTrainValues + expectedTestValues + expectedGroundTruthValues) * 4
        if data.count < expectedBytes {
            throw BenchmarkDatasetError.truncatedFile(expectedAtLeast: expectedBytes, actual: data.count)
        }

        var trainVectors: [[Float]] = []
        trainVectors.reserveCapacity(trainCount)
        for _ in 0..<trainCount {
            var vector: [Float] = []
            vector.reserveCapacity(dimension)
            for _ in 0..<dimension {
                vector.append(try readFloat32(from: data, cursor: &cursor))
            }
            trainVectors.append(vector)
        }

        var testVectors: [[Float]] = []
        testVectors.reserveCapacity(testCount)
        for _ in 0..<testCount {
            var vector: [Float] = []
            vector.reserveCapacity(dimension)
            for _ in 0..<dimension {
                vector.append(try readFloat32(from: data, cursor: &cursor))
            }
            testVectors.append(vector)
        }

        var groundTruth: [[UInt32]] = []
        groundTruth.reserveCapacity(testCount)
        for _ in 0..<testCount {
            var row: [UInt32] = []
            row.reserveCapacity(neighborsCount)
            for _ in 0..<neighborsCount {
                row.append(try readUInt32(from: data, cursor: &cursor))
            }
            groundTruth.append(row)
        }

        return BenchmarkDataset(
            trainVectors: trainVectors,
            testVectors: testVectors,
            groundTruth: groundTruth,
            dimension: dimension,
            metric: metric,
            neighborsCount: neighborsCount
        )
    }

    private func validate() throws {
        guard dimension > 0 else {
            throw BenchmarkDatasetError.invalidDataset("dimension must be > 0")
        }
        guard neighborsCount > 0 else {
            throw BenchmarkDatasetError.invalidDataset("neighborsCount must be > 0")
        }
        guard groundTruth.count == testVectors.count else {
            throw BenchmarkDatasetError.invalidDataset(
                "groundTruth row count \(groundTruth.count) must match test vector count \(testVectors.count)"
            )
        }
        for (index, vector) in trainVectors.enumerated() {
            guard vector.count == dimension else {
                throw BenchmarkDatasetError.invalidDataset(
                    "train vector \(index) has dimension \(vector.count), expected \(dimension)"
                )
            }
        }
        for (index, vector) in testVectors.enumerated() {
            guard vector.count == dimension else {
                throw BenchmarkDatasetError.invalidDataset(
                    "test vector \(index) has dimension \(vector.count), expected \(dimension)"
                )
            }
        }
        for (index, neighbors) in groundTruth.enumerated() {
            guard neighbors.count == neighborsCount else {
                throw BenchmarkDatasetError.invalidDataset(
                    "groundTruth row \(index) has count \(neighbors.count), expected \(neighborsCount)"
                )
            }
        }
    }

    private static func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    private static func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        k: Int,
        metric: Metric
    ) -> [UInt32] {
        vectors.enumerated()
            .map { (index, vector) in
                (UInt32(index), distance(query: query, vector: vector, metric: metric))
            }
            .sorted { lhs, rhs in
                lhs.1 < rhs.1
            }
            .prefix(k)
            .map(\.0)
    }

    private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        case .hamming:
            var mismatches = 0
            for d in 0..<query.count where query[d] != vector[d] {
                mismatches += 1
            }
            return Float(mismatches)
        }
    }

    private static func metricRaw(_ metric: Metric) -> UInt32 {
        switch metric {
        case .cosine:
            return 0
        case .l2:
            return 1
        case .innerProduct:
            return 2
        case .hamming:
            return 3
        }
    }

    private static func metric(fromRaw raw: UInt32) -> Metric? {
        switch raw {
        case 0:
            return .cosine
        case 1:
            return .l2
        case 2:
            return .innerProduct
        case 3:
            return .hamming
        default:
            return nil
        }
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        let little = value.littleEndian
        data.append(UInt8((little >> 0) & 0xFF))
        data.append(UInt8((little >> 8) & 0xFF))
        data.append(UInt8((little >> 16) & 0xFF))
        data.append(UInt8((little >> 24) & 0xFF))
    }

    private static func appendFloat32(_ value: Float, to data: inout Data) {
        appendUInt32(value.bitPattern, to: &data)
    }

    private static func readUInt32(from data: Data, cursor: inout Int) throws -> UInt32 {
        let next = cursor + 4
        if next > data.count {
            throw BenchmarkDatasetError.truncatedFile(expectedAtLeast: next, actual: data.count)
        }

        let b0 = UInt32(data[cursor])
        let b1 = UInt32(data[cursor + 1]) << 8
        let b2 = UInt32(data[cursor + 2]) << 16
        let b3 = UInt32(data[cursor + 3]) << 24
        cursor = next
        return b0 | b1 | b2 | b3
    }

    private static func readFloat32(from data: Data, cursor: inout Int) throws -> Float {
        Float(bitPattern: try readUInt32(from: data, cursor: &cursor))
    }
}
