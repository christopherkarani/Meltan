# MetalANNS Benchmark Report

Last updated: `2026-05-17`

## Purpose

This report captures the current benchmark numbers for evaluating MetalANNS as a replacement candidate for USearch. The focus is not only peak throughput, but whether the harness exposes the operational risks that matter for a production vector index: recall, tail latency, filtering, deletion, persistence, storage mode, streaming ingest, sharding, concurrency, GPU availability, and comparator availability.

## Environment

- Platform: `macOS`
- Build mode: `release`
- Benchmark target: `MetalANNSBenchmarks`
- Raw JSON output directory: `/tmp/metalanns-benchmarks-20260517`
- Estimated backend path for measured graph rows: `cpu`
- GPU parity mode: unavailable in this local build
- USearch comparison mode: unavailable in this local build

## Benchmark Configuration

Representative release workload:

| Setting | Value |
|---|---:|
| Vector count | `2000` |
| Dimension | `64` |
| Query count | `100` |
| Runs | `3` |
| Warmup runs | `1` |
| Degree | `32` |
| efSearch | `64` |
| Requested k | `10` |
| Effective search k | `100` |

The harness reports both requested and effective search depth because recall@100 forces deeper search than the user-facing `k=10` result set.

## Summary

| Mode | Recall@10 | QPS | Build / Op | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Single graph | 1.000 | 1614 | 411.5 ms build | 0.596 ms | 0.725 ms | 0.796 ms |
| IVFPQ graph baseline | 1.000 | 1577 | 402.2 ms build | 0.614 ms | 0.704 ms | 0.745 ms |
| IVFPQ | 0.963 | 5442 | 39.1 ms build | 0.165 ms | 0.210 ms | 0.273 ms |
| Sharded, 4 shards, nprobe 2 | 0.997 | 2879 | 194.7 ms build | 0.327 ms | 0.427 ms | 0.462 ms |
| Concurrent, 4 workers | 1.000 | 1587 | 416.5 ms build | 1.994 ms | 3.937 ms | 4.125 ms |
| Streaming ingest | n/a | 549 vec/s | 3640.6 ms ingest | n/a | n/a | n/a |
| Streaming flush | n/a | 0 | 422.5 ms flush | n/a | n/a | n/a |
| Streaming warm query | 1.000 | 1508 | n/a | 0.609 ms | 0.780 ms | 1.157 ms |
| Persistence save | n/a | 0 | 41.4 ms save | n/a | n/a | n/a |
| Persistence load | n/a | 0 | 34.3 ms load | n/a | n/a | n/a |
| Persistence first query | 1.000 | 1382 | 0.724 ms op | 0.724 ms | 0.724 ms | 0.724 ms |
| Persistence warm query | 1.000 | 1623 | n/a | 0.590 ms | 0.712 ms | 0.766 ms |

## Graph Baseline

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/single.json
```

| Metric | Value |
|---|---:|
| Build time | 411.5 ms |
| Query count | 300 |
| Requested k | 10 |
| Effective search k | 100 |
| Estimated backend | cpu |
| Query mean | 0.609 ms |
| Query p50 | 0.596 ms |
| Query p95 | 0.725 ms |
| Query p99 | 0.796 ms |
| QPS | 1614 |
| Recall@1 | 1.000 |
| Recall@10 | 1.000 |
| Recall@100 | 0.994 |

## IVFPQ

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --ivfpq \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --ivfpq-coarse-centroids 64 \
  --ivfpq-subspaces 8 \
  --ivfpq-nprobe 8 \
  --ivfpq-iterations 4 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/ivfpq.json
```

| Index | Recall@10 | QPS | Build | p50 | p95 | p99 | Estimated backend |
|---|---:|---:|---:|---:|---:|---:|---|
| `_GraphIndex` | 1.000 | 1577 | 402.2 ms | 0.614 ms | 0.704 ms | 0.745 ms | cpu |
| `_IVFPQIndex` | 0.963 | 5442 | 39.1 ms | 0.165 ms | 0.210 ms | 0.273 ms | ivfpq-adaptive |

Interpretation: IVFPQ is the fastest path in this run, with roughly `3.45x` graph-baseline QPS, but recall@10 drops from `1.000` to `0.963`.

## Filtering

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --filter-sweep \
  --filter-selectivity 0.5,0.1,0.01 \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/filter.json
```

| Filter selectivity | Recall@10 | QPS | Build | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 50% | 1.000 | 649 | 414.0 ms | 0.710 ms | 8.164 ms | 14.318 ms |
| 10% | 1.000 | 423 | 414.0 ms | 0.701 ms | 9.384 ms | 27.591 ms |
| 1% | 0.077 | 790 | 414.0 ms | 0.607 ms | 4.054 ms | 15.558 ms |

Interpretation: selective filtering is the clearest weak spot. At 1% selectivity, recall@10 collapses to `0.077`, which suggests filtering after ANN candidate generation can underfill true nearest-neighbor results.

## Deletion And Compaction

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --delete-sweep \
  --delete-ratios 0.1,0.3,0.5 \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/delete.json
```

| Mode | Recall@10 | QPS | Build | p50 | p95 | p99 | Operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% deleted, precompact | 1.000 | 1478 | 410.9 ms | 0.654 ms | 0.755 ms | 0.800 ms | 0.067 ms |
| 10% deleted, postcompact | 1.000 | 1765 | 410.9 ms | 0.539 ms | 0.678 ms | 0.740 ms | 525.6 ms compact |
| 30% deleted, precompact | 1.000 | 1246 | 412.8 ms | 0.766 ms | 0.936 ms | 1.053 ms | 0.128 ms |
| 30% deleted, postcompact | 1.000 | 882 | 412.8 ms | 0.434 ms | 5.540 ms | 13.282 ms | 523.7 ms compact |
| 50% deleted, precompact | 1.000 | 739 | 1188.9 ms | 0.890 ms | 2.955 ms | 10.709 ms | 0.172 ms |
| 50% deleted, postcompact | 1.000 | 2986 | 1188.9 ms | 0.318 ms | 0.371 ms | 0.413 ms | 515.4 ms compact |

Interpretation: recall stays stable in this synthetic run, but high tombstone ratios materially affect throughput before compaction. Compaction cost is roughly `515-526 ms` for this workload.

## Persistence

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --persistence \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/persistence.json
```

| Mode | Recall@10 | QPS | Build | p50 | p95 | p99 | Operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Save | n/a | 0 | 408.6 ms | n/a | n/a | n/a | 41.4 ms |
| Load | n/a | 0 | 408.6 ms | n/a | n/a | n/a | 34.3 ms |
| First query after load | 1.000 | 1382 | 408.6 ms | 0.724 ms | 0.724 ms | 0.724 ms | 0.724 ms |
| Warm query after load | 1.000 | 1623 | 408.6 ms | 0.590 ms | 0.712 ms | 0.766 ms | n/a |

Interpretation: save/load costs are small on this workload, and loaded-query recall remains intact.

## Storage Modes

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --storage-sweep \
  --storage-modes normal,mmap,disk-backed \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/storage.json
```

| Mode | Recall@10 | QPS | Build | p50 | p95 | p99 | Operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal load | n/a | 0 | 419.6 ms | n/a | n/a | n/a | 25.4 ms |
| Normal first query | 1.000 | 1603 | 419.6 ms | 0.624 ms | 0.624 ms | 0.624 ms | 0.624 ms |
| Normal warm query | 1.000 | 1548 | 419.6 ms | 0.610 ms | 0.778 ms | 0.924 ms | n/a |
| Disk-backed load | n/a | 0 | 419.6 ms | n/a | n/a | n/a | 28.3 ms |
| Disk-backed warm query | 1.000 | 959 | 419.6 ms | 0.998 ms | 1.218 ms | 1.381 ms | n/a |
| Normal save | n/a | 0 | 419.6 ms | n/a | n/a | n/a | 33.2 ms |
| mmap save v3 | n/a | 0 | 419.6 ms | n/a | n/a | n/a | 16.7 ms |
| mmap load | n/a | 0 | 419.6 ms | n/a | n/a | n/a | 28.1 ms |
| mmap first query | 1.000 | 1601 | 419.6 ms | 0.625 ms | 0.625 ms | 0.625 ms | 0.625 ms |
| mmap warm query | 1.000 | 1442 | 419.6 ms | 0.647 ms | 0.784 ms | 0.896 ms | n/a |

Interpretation: disk-backed query throughput is materially weaker than normal and mmap modes on this workload: `959 QPS` versus `1548 QPS` normal warm query and `1442 QPS` mmap warm query.

## Streaming

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --streaming \
  --streaming-batch-size 128 \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/streaming.json
```

| Mode | Recall@10 | QPS | Build | p50 | p95 | p99 | Operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streaming ingest | n/a | 549 vec/s | n/a | n/a | n/a | n/a | 3640.6 ms |
| Streaming flush | n/a | 0 | n/a | n/a | n/a | n/a | 422.5 ms |
| Streaming warm query | 1.000 | 1508 | n/a | 0.609 ms | 0.780 ms | 1.157 ms | n/a |

Interpretation: warm-query recall is good after streaming ingest and flush, but ingest cost is significant at this size. Sustained mixed read/write workloads should remain a release gate.

## Sharding

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --sharded \
  --shards 4 \
  --nprobe 2 \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/sharded.json
```

| Mode | Recall@10 | QPS | Build | p50 | p95 | p99 | Operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sharded build | n/a | 0 | 194.7 ms | n/a | n/a | n/a | 194.7 ms |
| Sharded search | 0.997 | 2879 | 194.7 ms | 0.327 ms | 0.427 ms | 0.462 ms | n/a |

Interpretation: sharding is strong in this run: nearly `1.78x` the single graph QPS with only a small recall@10 drop from `1.000` to `0.997`.

## Concurrent Search

Command:

```bash
swift run -c release MetalANNSBenchmarks \
  --concurrent \
  --concurrency 4 \
  --vector-count 2000 \
  --dimension 64 \
  --query-count 100 \
  --degree 32 \
  --efsearch 64 \
  --k 10 \
  --runs 3 \
  --warmup 1 \
  --json-out /tmp/metalanns-benchmarks-20260517/concurrent.json
```

| Mode | Recall@10 | QPS | Build | p50 | p95 | p99 | Estimated backend |
|---|---:|---:|---:|---:|---:|---:|---|
| Concurrent search, 4 workers | 1.000 | 1587 | 416.5 ms | 1.994 ms | 3.937 ms | 4.125 ms | cpu |

Interpretation: recall holds, but four-way concurrency does not materially exceed the single graph path and increases per-query latency. This is a scaling weakness to investigate before replacement claims.

## Comparator And GPU Parity Availability

Commands:

```bash
swift run -c release MetalANNSBenchmarks \
  --gpu-parity \
  --json-out /tmp/metalanns-benchmarks-20260517/gpu-parity.json

swift run -c release MetalANNSBenchmarks \
  --usearch-compare \
  --json-out /tmp/metalanns-benchmarks-20260517/usearch.json
```

| Mode | Status |
|---|---|
| GPU parity | unavailable |
| USearch comparison | unavailable |

Interpretation: the current local build produced deterministic unavailable rows rather than direct GPU parity or USearch comparator numbers. That is acceptable harness behavior, but not sufficient evidence for a USearch replacement decision.

## Weak Areas Exposed

1. Selective filtering is the highest-priority correctness risk.
   - At 1% selectivity, recall@10 fell to `0.077`.
   - The harness now makes this visible instead of hiding it behind unfiltered synthetic results.

2. Concurrent search does not scale on this workload.
   - Single graph: `1614 QPS`.
   - Four-worker concurrent search: `1587 QPS`, with p50 rising to `1.994 ms`.

3. Disk-backed query mode is slower than normal and mmap modes.
   - Disk-backed warm query: `959 QPS`.
   - Normal warm query: `1548 QPS`.
   - mmap warm query: `1442 QPS`.

4. IVFPQ speed comes with a measurable recall tradeoff.
   - IVFPQ: `5442 QPS`, recall@10 `0.963`.
   - Graph baseline: about `1577-1614 QPS`, recall@10 `1.000`.

5. Direct replacement evidence is incomplete without live USearch comparison.
   - The harness can emit an explicit unavailable row, but a replacement decision still needs same-input USearch numbers.

6. GPU parity is not currently proven by this run.
   - The measured graph rows report estimated CPU path.
   - GPU parity mode reported unavailable in this local build.

## Replacement Readiness Readout

MetalANNS is promising on synthetic release workloads, especially with IVFPQ and sharding. It is not yet proven as a USearch replacement from this run alone.

The minimum next evidence needed:

- Direct USearch side-by-side comparison on the same frozen datasets.
- Real embedding datasets, not only synthetic vectors.
- Forced CPU/GPU parity runs for graph and IVFPQ paths.
- Filtering fixes or query planning changes for highly selective predicates.
- Sustained concurrent and mixed read/write benchmarks.
- Larger scale sweeps across vector count, dimension, k, efSearch, shard count, nprobe, and storage modes.

## Reproduce

```bash
mkdir -p /tmp/metalanns-benchmarks-20260517

swift run -c release MetalANNSBenchmarks --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/single.json

swift run -c release MetalANNSBenchmarks --ivfpq --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --ivfpq-coarse-centroids 64 --ivfpq-subspaces 8 --ivfpq-nprobe 8 --ivfpq-iterations 4 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/ivfpq.json

swift run -c release MetalANNSBenchmarks --filter-sweep --filter-selectivity 0.5,0.1,0.01 --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/filter.json

swift run -c release MetalANNSBenchmarks --delete-sweep --delete-ratios 0.1,0.3,0.5 --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/delete.json

swift run -c release MetalANNSBenchmarks --persistence --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/persistence.json

swift run -c release MetalANNSBenchmarks --storage-sweep --storage-modes normal,mmap,disk-backed --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/storage.json

swift run -c release MetalANNSBenchmarks --streaming --streaming-batch-size 128 --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/streaming.json

swift run -c release MetalANNSBenchmarks --sharded --shards 4 --nprobe 2 --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/sharded.json

swift run -c release MetalANNSBenchmarks --concurrent --concurrency 4 --vector-count 2000 --dimension 64 --query-count 100 --degree 32 --efsearch 64 --k 10 --runs 3 --warmup 1 --json-out /tmp/metalanns-benchmarks-20260517/concurrent.json

swift run -c release MetalANNSBenchmarks --gpu-parity --json-out /tmp/metalanns-benchmarks-20260517/gpu-parity.json

swift run -c release MetalANNSBenchmarks --usearch-compare --json-out /tmp/metalanns-benchmarks-20260517/usearch.json
```
