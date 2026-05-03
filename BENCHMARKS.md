# MetalANNS Benchmarks

Last updated: `2026-03-07`

## Environment

- Architecture: `arm64`
- Platform: `macOS`
- Runtime note: Metal shader loading is now validated in this environment via fallback library loading and bundled-source compilation.

## Benchmark Configuration

Primary synthetic benchmark configuration:

- Vector count: `1000`
- Dimension: `128`
- Degree: `32`
- Query count: `200`
- Search `k`: `10`
- Effective benchmark search depth: top-`100`
- Default `efSearch`: `64`
- Metrics exercised: `cosine`, `l2`

## Current Measured Results

Clean isolated release runs:

### GraphIndex

Command:

```bash
swift run -c release MetalANNSBenchmarks --query-count 200 --runs 3 --warmup 1
```

| Metric | Value |
|---|---:|
| Build time (ms) | 215.3 |
| Query mean (ms) | 0.315 |
| Query p50 (ms) | 0.30 |
| Query p95 (ms) | 0.38 |
| Query p99 (ms) | 0.47 |
| Query QPS | 3073.26 |
| Recall@1 | 1.000 |
| Recall@10 | 1.000 |
| Recall@100 | 1.000 |

### GraphIndex vs IVFPQ

Command:

```bash
swift run -c release MetalANNSBenchmarks --ivfpq --query-count 200 --runs 3 --warmup 1
```

| Index | Build (ms) | QPS | p50 (ms) | p95 (ms) | p99 (ms) | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| `_GraphIndex` | 175.2 | 3349 | 0.29 | 0.30 | 0.32 | 1.000 |
| `_IVFPQIndex` | 36.2 | 6657 | 0.14 | 0.16 | 0.17 | 0.995 |

## Debug Perf Suites

Measured with focused Swift Testing filters:

| Suite | Current Result |
|---|---:|
| `IVFPQComprehensiveTests.benchmarkSearchThroughput` | `327.34 qps` |
| `IVFPQComprehensiveTests.benchmarkRecallVsQPS` runtime | `1.30 s` |
| `ShardedIndexParallelBuildTests.parallelBuildTimingLogged` speedup | `2.23x` |
| `ShardedIndexParallelSearchTests.parallelSearchTimingLogged` parallel QPS | `346.46` |

## Improvement Multiples

Compared against the original baselines recorded before the performance remediation:

| Area | Before | After | Improvement |
|---|---:|---:|---:|
| `_IVFPQIndex` release build | `3225.2 ms` | `36.2 ms` | `89.1x faster` |
| `_IVFPQIndex` release QPS | `4260` | `6657` | `1.56x faster` |
| `_IVFPQIndex` release recall@10 | `0.965` | `0.995` | `1.03x higher` |
| IVFPQ comprehensive throughput | `202.42 qps` | `327.34 qps` | `1.62x faster` |
| IVFPQ recall-vs-QPS runtime | `58.79 s` | `1.30 s` | `45.2x faster` |
| Sharded parallel build time | `0.3333 s` | `0.2318 s` | `1.44x faster` |
| Sharded build speedup ratio | `1.88x` | `2.23x` | `1.19x better` |
| Sharded parallel search QPS | `319.42` | `346.46` | `1.08x faster` |
| `_GraphIndex` build in IVFPQ harness | `193.9 ms` | `175.2 ms` | `1.11x faster` |
| `_GraphIndex` QPS in IVFPQ harness | `3224` | `3349` | `1.04x faster` |

Additional note: during tuning, an intermediate small-workload GPU dispatch regression dropped `_GraphIndex` to `47.0 QPS`. The final workload-aware CPU/GPU gating restored that path to `3073.26 QPS`, which is a `65.4x` recovery from the bad intermediate state.

## Main Performance Changes Behind The Gains

- KMeans now parallelizes Lloyd iterations safely and avoids subspace-materialization overhead.
- Graph pruning uses a pointer fast path for `VectorBuffer` instead of repeated vector extraction.
- GPU search and GPU ADC reuse Metal workspaces instead of allocating fresh buffers per expansion/query.
- IVFPQ add/training/search removed major allocation, sorting, and serial planning overhead.
- `_GraphIndex` now routes small builds and small searches to CPU paths where GPU submission overhead loses.
- Metal shader library loading now has robust fallbacks, so GPU paths benchmark correctly in this environment.

## Reproduce

```bash
swift build
swift test --filter MetalDeviceTests
swift test --filter MetalSearchTests
swift test --filter FullGPUSearchTests
swift test --filter IVFPQComprehensiveTests.benchmarkSearchThroughput
swift test --filter IVFPQComprehensiveTests.benchmarkRecallVsQPS
swift run -c release MetalANNSBenchmarks --query-count 200 --runs 3 --warmup 1
swift run -c release MetalANNSBenchmarks --ivfpq --query-count 200 --runs 3 --warmup 1
```

## Harness Capabilities

The benchmark CLI emits a startup banner (Metal device, OS build, core count, thermal/low-power state) and per-run captures latency distribution, memory snapshots, and cold-vs-warm timings.

Latency reporting:
- Per-query latencies are pooled across runs and percentiles (P50/P90/P95/P99/P99.9), mean, stddev, min, max are computed from the pool.
- An ASCII histogram is printed at the end of every single run.
- `--histogram-out <path>` writes `<path>.histogram.csv` and `<path>.cdf.csv`.

Cold-vs-warm:
- The very first dispatch after build is timed separately as `firstQueryLatencyMs` (cold).
- `warmSteadyMeanMs` is the mean of all subsequent measured queries, excluding the first.

Memory:
- `MemorySnapshot.capture()` records `phys_footprint` / `resident_size` / `resident_size_peak` via `task_info` before and after build and after queries.
- Each row reports `indexResidentMB` (post-build delta) and `peakResidentMB`.

Sweeps:
- `--sweep` sweeps `efSearch`. Combine with `--sweep-degree D1,D2,...` to do a 2D `degree × efSearch` cross-product. Each row is labeled `degree=D,efSearch=E`.
- `--concurrency-sweep 1,2,4,8,16` sweeps in-flight query count via a `TaskGroup` sliding window through `_GraphIndex.search`. One row per level, labeled `concurrency=N`.
- After any sweep, the table is followed by an ASCII Pareto chart (recall@10 vs log10 QPS) with `*` for frontier points and `.` for dominated points.

Other flags worth knowing:
- `--concurrency N` runs a single benchmark with `N` in-flight queries (not a sweep).
- `--compare cpu,gpu,gpu-adc` emits one row per backend label. NOTE: `_GraphIndex` does not currently expose a public backend selector — the harness prints a warning at startup; the per-label rows reflect run-to-run variance of the auto-selected backend, not distinct backends. Wire-up is structured so a real selector is a one-line swap.
- `--csv-out <path>` writes a wide CSV with all latency, recall, memory, and concurrency fields.
- `--json-out <path>` writes the same fields plus full environment metadata.

Examples:

```bash
# Single run with histogram + cold/warm breakdown
swift run -c release MetalANNSBenchmarks \
    --dataset sift1m.annbin --runs 5 --warmup 2 --histogram-out reports/sift1m

# 2D Pareto sweep over (degree, efSearch)
swift run -c release MetalANNSBenchmarks \
    --dataset sift1m.annbin --sweep \
    --sweep-degree 16,32,48 --sweep-efsearch 16,32,64,128,256 \
    --csv-out reports/sift1m-2d.csv

# Throughput-vs-latency curve
swift run -c release MetalANNSBenchmarks \
    --dataset sift1m.annbin --concurrency-sweep 1,2,4,8,16,32 \
    --csv-out reports/sift1m-conc.csv
```
