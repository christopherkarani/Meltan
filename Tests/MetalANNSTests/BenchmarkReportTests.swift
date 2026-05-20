import Foundation
import Testing
@testable import MetalANNSBenchmarks

@Suite("BenchmarkReport Tests")
struct BenchmarkReportTests {
    @Test("tableOutput")
    func tableOutput() {
        let report = sampleReport()
        let table = report.renderTable()

        #expect(table.contains("label"))
        #expect(table.contains("recall@10"))
        #expect(table.contains("QPS"))
        #expect(table.contains("efSearch=16"))
        #expect(table.contains("efSearch=128"))
    }

    @Test("csvOutput")
    func csvOutput() {
        let report = sampleReport()
        let csv = report.renderCSV()
        let lines = csv.split(separator: "\n", omittingEmptySubsequences: true)

        #expect(lines.count == 4)
        #expect(String(lines[0]).contains("effectiveK"))
        #expect(String(lines[0]).contains("queryCount"))
        #expect(String(lines[0]).contains("operationTimeMs"))
        #expect(String(lines[0]).contains("estimatedBackendPath"))
        #expect(String(lines[0]).contains("rssBeforeBytes"))
        #expect(String(lines[0]).contains("fileSizeBytes"))
    }

    @Test("jsonIncludesEffectiveDepthAndMetadata")
    func jsonIncludesEffectiveDepthAndMetadata() {
        let report = BenchmarkReport(
            rows: [
                .init(
                    label: "single",
                    recallAt10: 0.9,
                    qps: 1000,
                    buildTimeMs: 50,
                    p50Ms: 1,
                    p95Ms: 2,
                    p99Ms: 3,
                    requestedK: 10,
                    effectiveK: 100,
                    estimatedBackendPath: "cpu",
                    rssBeforeBytes: 100,
                    rssAfterBytes: 140,
                    rssDeltaBytes: 40,
                    fileSizeBytes: 4096
                )
            ],
            datasetLabel: "synthetic",
            metadata: ["gitCommit": "abc123", "datasetHash": "hash"]
        )

        let json = report.renderJSON()

        #expect(json.contains("\"requestedK\""))
        #expect(json.contains("\"effectiveK\""))
        #expect(json.contains("\"operationTimeMs\""))
        #expect(json.contains("\"estimatedBackendPath\""))
        #expect(json.contains("\"rssBeforeBytes\""))
        #expect(json.contains("\"fileSizeBytes\""))
        #expect(json.contains("\"gitCommit\""))
        #expect(json.contains("\"datasetHash\""))
    }

    @Test("rowsCanRepresentLifecycleWorkloads")
    func rowsCanRepresentLifecycleWorkloads() {
        let report = BenchmarkReport(
            rows: [
                .init(
                    label: "filter=0.100",
                    recallAt10: 0.85,
                    qps: 750,
                    buildTimeMs: 100,
                    p50Ms: 1,
                    p95Ms: 2,
                    p99Ms: 3,
                    requestedK: 10,
                    effectiveK: 100,
                    operation: "filter",
                    operationTimeMs: 4,
                    estimatedBackendPath: "filtered-cpu",
                    rssBeforeBytes: 100,
                    rssAfterBytes: 140,
                    rssDeltaBytes: 40,
                    fileSizeBytes: 4096
                ),
                .init(
                    label: "persistence=load",
                    recallAt10: 1,
                    qps: 0,
                    buildTimeMs: 100,
                    p50Ms: 0,
                    p95Ms: 0,
                    p99Ms: 0,
                    operation: "load",
                    operationTimeMs: 10,
                    estimatedBackendPath: "persistence"
                )
            ],
            datasetLabel: "synthetic",
            metadata: ["mode": "filter-sweep"]
        )

        let csv = report.renderCSV()
        let json = report.renderJSON()

        #expect(csv.contains("filter=0.100"))
        #expect(csv.contains("persistence=load"))
        #expect(json.contains("\"mode\""))
        #expect(json.contains("\"operation\""))
        #expect(json.contains("\"load\""))
        #expect(json.contains("\"operationTimeMs\""))
        #expect(json.contains("\"rssDeltaBytes\""))
        #expect(json.contains("\"fileSizeBytes\""))
        #expect(json.contains("filtered-cpu"))
    }

    @Test("paretoFrontier")
    func paretoFrontier() {
        let rows: [BenchmarkReport.Row] = [
            .init(label: "a", recallAt10: 0.90, qps: 4000, buildTimeMs: 100, p50Ms: 1, p95Ms: 2, p99Ms: 3),
            .init(label: "b", recallAt10: 0.92, qps: 3500, buildTimeMs: 110, p50Ms: 1.1, p95Ms: 2.1, p99Ms: 3.1),
            .init(label: "c", recallAt10: 0.91, qps: 3000, buildTimeMs: 120, p50Ms: 1.2, p95Ms: 2.2, p99Ms: 3.2), // dominated by b
            .init(label: "d", recallAt10: 0.95, qps: 2500, buildTimeMs: 130, p50Ms: 1.3, p95Ms: 2.3, p99Ms: 3.3),
            .init(label: "e", recallAt10: 0.88, qps: 3900, buildTimeMs: 90, p50Ms: 0.9, p95Ms: 1.9, p99Ms: 2.9) // dominated by a
        ]

        let frontier = BenchmarkReport(rows: rows, datasetLabel: "synthetic").paretoFrontier()
        let labels = Set(frontier.map(\.label))

        #expect(frontier.count == 3)
        #expect(labels.contains("a"))
        #expect(labels.contains("b"))
        #expect(labels.contains("d"))
        #expect(!labels.contains("c"))
        #expect(!labels.contains("e"))
    }
}

private func sampleReport() -> BenchmarkReport {
    BenchmarkReport(
        rows: [
            .init(label: "efSearch=16", recallAt10: 0.841, qps: 6800, buildTimeMs: 141.8, p50Ms: 0.8, p95Ms: 1.4, p99Ms: 2.2, requestedK: 10, effectiveK: 100, estimatedBackendPath: "cpu"),
            .init(label: "efSearch=64", recallAt10: 0.953, qps: 4231, buildTimeMs: 142.0, p50Ms: 1.2, p95Ms: 2.1, p99Ms: 3.4, requestedK: 10, effectiveK: 100, estimatedBackendPath: "cpu"),
            .init(label: "efSearch=128", recallAt10: 0.971, qps: 2108, buildTimeMs: 142.0, p50Ms: 2.4, p95Ms: 4.2, p99Ms: 6.8, requestedK: 10, effectiveK: 100, estimatedBackendPath: "cpu")
        ],
        datasetLabel: "synthetic"
    )
}
