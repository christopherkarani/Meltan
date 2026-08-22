# MetalANNS Benchmarks

Last updated: `2026-08-22`

## 2026-08 — Production Acceptance Matrix (10k–200k × cosine/dot/L2)

Full release-mode acceptance run against the legacy Metal graph-search path
(the same-binary A/B via `--exact-search-limit 0`, which restores the
per-graph-hop GPU dispatch behavior that a Wax-style legacy Metal vector
search uses). All runs are seeded (`--seed 42`) and reproducible: dim 128,
k=10 (recall@100 also tracked), 200 queries, concurrency 1 unless noted.

### Same-binary A/B vs legacy path (concurrency 1, M3 Max, release)

| Corpus | Metric | Legacy QPS | Exact QPS | QPS speedup | Legacy p50 | Exact p50 | p50 speedup | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10k  | cosine | 303 | 10,269 | **33.9x** | 3.29 ms | 0.085 ms | **38.8x** | 1.000 |
| 10k  | dot    | 302 | 11,062 | **36.6x** | 3.28 ms | 0.078 ms | **42.2x** | 1.000 |
| 10k  | L2     | 317 | 10,666 | **33.7x** | 3.13 ms | 0.081 ms | **38.5x** | 1.000 |
| 50k  | cosine | 61 | 2,470 | **40.6x** | 16.34 ms | 0.330 ms | **49.6x** | 1.000 |
| 50k  | dot    | 63 | 2,415 | **38.6x** | 15.98 ms | 0.341 ms | **46.9x** | 1.000 |
| 50k  | L2     | 60 | 2,459 | **41.1x** | 16.62 ms | 0.333 ms | **50.0x** | 1.000 |
| 100k | cosine | 27 | 1,455 | **54.0x** | 36.34 ms | 0.577 ms | **63.0x** | 1.000 |
| 100k | dot    | 32 | 1,683 | **53.1x** | 30.40 ms | 0.498 ms | **61.1x** | 1.000 |
| 100k | L2     | 29 | 1,704 | **59.3x** | 34.63 ms | 0.491 ms | **70.5x** | 1.000 |
| 200k | cosine | 14 | 1,167 | **83.9x** | 71.38 ms | 0.758 ms | **94.2x** | 1.000 |
| 200k | dot    | 18 | 1,150 | **65.4x** | 56.62 ms | 0.763 ms | **74.2x** | 0.9995* |
| 200k | L2     | 14 | 1,046 | **74.2x** | 70.80 ms | 0.807 ms | **87.7x** | 1.000 |

Minimum speedup across the entire matrix: **33.7x QPS / 38.5x warm p50** —
every cell exceeds the 10x target by 3x or more. The legacy path also loses
recall at scale (dot @ 100k: recall@10 = 0.483; cosine @ 200k prior runs:
0.768) while the exact path holds recall@k = 1.0 by construction.
*The single 0.9995 cell is a distance-tie at the k-th boundary in the seeded
synthetic data (brute-force sort and heap selector break ties differently);
re-running with `--seed 43` yields 1.000/1.000.

### QPS under concurrency (exact path, cosine)

| Corpus | c=1 | c=8 | c=16 | Recall@10 (all levels) |
|---|---:|---:|---:|---:|
| 10k  | 10,407 | 20,591 | 13,388 | 1.000 |
| 50k  | 2,113 | 9,859 | 11,873 | 1.000 |
| 100k | 1,507 | 4,446 | 4,594 | 1.000 |
| 200k | 1,033 | 2,244 | 2,259 | 1.000 |

Cross-metric at 50k: dot 2,195 / 10,873 / 10,491 and L2 2,180 / 11,001 /
11,853 QPS at c=1/8/16, recall 1.000 throughout. 10k saturates the host-tier
BLAS scan near c=8 (CPU-bound); 100k+ saturate the single GPU's dispatch
pipeline — in all cases ≥33x over the legacy path at the same concurrency.

### Cold vs warm (exact path, first dispatch after build vs steady state)

| Corpus | Cold (first query) | Warm steady mean |
|---|---:|---:|
| 10k  | 0.2–0.95 ms | 0.081–0.087 ms |
| 50k  | 0.9–2.1 ms | 0.395–0.404 ms |
| 100k | 1.3–1.9 ms | 0.576–0.671 ms |
| 200k | 1.4–2.0 ms | 0.846–0.944 ms |

Cold penalty is bounded (~2x warm at the largest scale) — no lazy pipeline
compilation cliff after the first query.

### Build, update, and persistence costs (cosine, `--measure-updates --update-count 200`)

| Corpus | Build | Insert | Delete | Compact | Save | Load | Recall after ins/del/compact/reload |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10k  | 1.6 s | 1.37 ms/vec | ~0 ms | 1.6 s | 49 ms | 136 ms | 1.0 / 1.0 / 1.0 / 1.0 |
| 50k  | 2.4 s | 1.68 ms/vec | ~0 ms | 2.3 s | 163 ms | 0.92 s | 1.0 / 1.0 / 1.0 / 1.0 |
| 100k | 4.8 s | 1.83 ms/vec | ~0 ms | 5.0 s | 327 ms | 2.2 s | 1.0 / 1.0 / 1.0 / 1.0 |
| 200k | 10.9 s | 1.79 ms/vec | ~0 ms | 10.4 s | 0.71 s | 5.1 s | 1.0 / 1.0 / 1.0 / 1.0 |

Deletes are O(1) soft deletes. Compact costs ≈ one rebuild (it rebuilds
storage without the deleted rows) and restores the fully-compacted exact
path. Memory: index resident set 221 MB (10k) → 507 MB (200k cosine,
~2.5 KB/vector including graph adjacency).

### Changes behind this round

- **Delete-aware exact routing** (`ANNSIndex+Search.swift`): the fused exact
  path previously bailed to the heuristic graph traversal whenever any row
  was soft-deleted. Scanning top-(k + deletedCount) provably keeps ≥ k
  survivors, so single-query and `batchSearch` now over-fetch by the
  deletion count and filter afterwards — post-delete search stays exact and
  fast (previously: graph recall 0.8–0.99 at 1/30th the speed). Heavy
  deletion counts (> 256 − k) still route to the graph path.
- **Update-cost benchmarking** (`--measure-updates`, `--update-count`):
  measures insert/delete/compact/save/load plus recall after each mutation
  against an independent BLAS ground truth computed harness-side.
- **Exact-path integration tests** (`ExactPathIntegrationTests`, 10 tests):
  recall@k = 1.0 across metrics on host and GPU tiers (the GPU kernel tier
  previously had no direct test coverage), fallback when
  `exactSearchMaxVectorCount = 0` and for float16 storage, delete filtering
  with exact post-delete recall (single + batch), insert/norm-cache
  invalidation, compaction exactness, and persistence round-trip.

### Reproduce

```bash
swift build -c release
# One A/B pair:
.build/release/MetalANNSBenchmarks --vector-count 100000 --query-count 200 \
    --runs 3 --warmup 1 --seed 42 --metric cosine --csv-out exact.csv
.build/release/MetalANNSBenchmarks --vector-count 100000 --query-count 100 \
    --runs 1 --seed 42 --metric cosine --exact-search-limit 0 --csv-out legacy.csv
# Concurrency sweep and update costs:
.build/release/MetalANNSBenchmarks --vector-count 100000 --query-count 200 \
    --runs 2 --warmup 1 --seed 42 --concurrency-sweep 1,8,16 --csv-out conc.csv
.build/release/MetalANNSBenchmarks --vector-count 100000 --query-count 200 \
    --seed 42 --measure-updates --update-count 200 --csv-out upd.csv
```

## 2026-08 — Fused Exact Search (FlatGPUSearch)

Vector search retrieval was re-plumbed around a fused exact top-K path that
replaces per-graph-hop GPU dispatches (one command-buffer commit + sync per
beam-search iteration) with either a single-dispatch Metal kernel or a host
BLAS scan:

- `flat_scan_distances` (`Sources/MetalANNSCore/Shaders/FlatSearch.metal`):
  one thread per (query, row), float4 loads, no barriers; all metrics reduce
  to `dot(q,v)` + `||v||²`.
- Host tier (`FlatGPUSearch.hostSearch`): `cblas_sgemv` dot pass + cached
  corpus norms + bounded max-heap top-K. Wins below ~32k vectors where a GPU
  dispatch round trip costs more than the scan itself.
- GPU tier: one command buffer for the whole scan, host-side selection.
- Norm cache is invalidated on in-place storage writes (`VectorBuffer`
  insert hooks), so results remain exact.
- Gated by `IndexConfiguration.exactSearchMaxVectorCount` (0 disables).

### Same-binary A/B (M3 Max, macOS 26.0, release, 200 queries, 3 runs)

`--exact-search-limit 0` disables the fast path and restores legacy behavior,
so both columns below come from the same binary.

| Corpus | Baseline QPS | Baseline p50 | Exact QPS | Exact p50 | Speedup | Recall@10 |
|---|---:|---:|---:|---:|---:|---|
| 1k vectors   | 2,850  | 0.34 ms  | 24,928 | 0.029 ms | **8.7x**  | 1.000 |
| 8k vectors   | 372    | 2.68 ms  | 11,795 | 0.075 ms | **31.7x** | 1.000 |
| 16k vectors  | 184    | 5.43 ms  | 7,730  | 0.119 ms | **42.1x** | 1.000 |
| 50k vectors  | 57     | 17.4 ms  | 2,216  | 0.441 ms | **38.5x** | 1.000 |
| 200k vectors | 13.6   | 73.4 ms  | 1,042  | 0.947 ms | **76.6x** | 1.000 |

Geometric-mean speedup across scales: **~32x**. The legacy graph path also
loses recall at 200k vectors (recall@10 = 0.768, recall@100 = 0.792) while
the exact path holds recall@k = 1.000 by construction.

### Throughput under concurrency and batching (50k vectors, cosine)

| Mode | QPS |
|---|---:|
| Legacy baseline, sequential | 57 |
| Exact path, concurrency=8   | 11,521 |
| Exact path, concurrency=16  | 12,752 |
| `batchSearch`, 200 queries/dispatch | 7,136 |

At 200k vectors the batched API sustains 1,547 QPS (~114x baseline).

### Notes

- Dispatch latency on this machine/environment carries a large fixed tax
  (~200-350 us per command buffer under `powermode=Automatic`), which is why
  the hybrid beam-search design collapsed as graphs deepened and why small
  corpora are served from the host tier.
- Full methodology, kernel-level profiling narrative, and charts:
  `docs/performance-report.html`.

## Historical Results

Last updated before 2026-08: `2026-03-07`

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
