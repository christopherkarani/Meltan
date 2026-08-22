# MetalANNS Benchmarks

Last updated: `2026-08-22`

## 2026-08-22 — Production acceptance run (fused exact search, full matrix)

Release-mode verification of the fused exact-search path against the legacy
graph-traversal baseline (same binary, `--exact-search-limit 0` restores the
legacy path). Environment: Apple M3 Max, macOS 26.0 (25A354), release build,
thermal nominal, seeded synthetic data (`--seed 42`, dim 128, degree 32,
efSearch 64), 200 queries, k measured over top-100 ground truth,
`--runs 3 --warmup 1`.

### Warm latency + QPS by scale and metric (concurrency = 1)

| Metric | Corpus | Exact QPS | Exact p50 | Exact p95 | Legacy QPS | Legacy p50 | Legacy p95 | Speedup | Recall@10 exact / legacy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cosine | 10k   | 9,393  | 0.09 ms | 0.11 ms | 343   | 2.89 ms  | 3.09 ms  | **27.4x** | 1.000 / 1.000 |
| cosine | 50k   | 2,450  | 0.34 ms | 0.80 ms | 66    | 14.94 ms | 15.90 ms | **36.9x** | 1.000 / 1.000 |
| cosine | 100k  | 1,544  | 0.52 ms | 1.16 ms | 32    | 31.00 ms | 31.72 ms | **47.9x** | 1.000 / 0.989 |
| cosine | 200k  | 1,159  | 0.76 ms | 1.33 ms | 15.8  | 63.26 ms | 64.71 ms | **73.4x** | 1.000 / 0.802 |
| l2     | 10k   | 9,511  | 0.09 ms | 0.11 ms | 342   | 2.90 ms  | 3.13 ms  | **27.8x** | 1.000 / 1.000 |
| l2     | 50k   | 1,997  | 0.36 ms | 1.16 ms | 65    | 15.23 ms | 16.01 ms | **30.5x** | 1.000 / 1.000 |
| l2     | 100k  | 1,728  | 0.49 ms | 1.01 ms | 32    | 31.09 ms | 32.02 ms | **53.9x** | 1.000 / 0.988 |
| l2     | 200k  | 1,127  | 0.79 ms | 1.36 ms | 15.8  | 63.21 ms | 65.16 ms | **71.5x** | 1.000 / 0.805 |
| dot    | 10k   | 9,745  | 0.09 ms | 0.11 ms | 333   | 2.99 ms  | 3.18 ms  | **29.3x** | 1.000 / 1.000 |
| dot    | 50k   | 2,442  | 0.34 ms | 0.75 ms | 69    | 14.36 ms | 15.04 ms | **35.2x** | 1.000 / 0.693 |
| dot    | 100k  | 1,536  | 0.54 ms | 1.14 ms | 36    | 27.48 ms | 28.52 ms | **42.4x** | 1.000 / 0.481 |
| dot    | 200k  | 1,116  | 0.79 ms | 1.38 ms | 19.4  | 51.48 ms | 53.15 ms | **57.6x** | 1.000 / 0.150 |

Geometric-mean speedup across all 12 cells: **~41x**. The exact path holds
recall@10 = recall@100 = 1.000 in every cell; the legacy graph path loses
recall as graphs deepen (0.15 at 200k dot-product) while also running 16-74x
slower.

### Throughput under concurrency (cosine)

| Corpus | Path | c=1 | c=8 | c=16 |
|---|---|---:|---:|---:|
| 50k  | exact  | 2,251 QPS | 11,047 QPS | 11,941 QPS |
| 50k  | legacy | 64 QPS    | 71 QPS     | -          |
| 200k | exact  | 1,017 QPS | 2,353 QPS  | 2,286 QPS  |
| 200k | legacy | 15.8 QPS  | 16.6 QPS   | -          |

Steady-state QPS advantage at concurrency 8: **156x** (50k) and **142x**
(200k). Recall stays 1.000 for the exact path at every concurrency level.

### Cold vs warm

- Exact path cold first query: 0.9-3.2 ms across scales (pipeline compile +
  first dispatch); warm steady means 0.09 ms (10k) to 0.85 ms (200k).
- Legacy path cold first query: 85 ms (10k) to 4.7 s (200k) due to lazy HNSW
  construction on first search; warm steady means 2.9 ms to 63 ms.

### Memory (post-build resident delta / peak resident after queries)

| Corpus | Index resident | Peak (exact) | Peak (legacy) |
|---|---:|---:|---:|
| 50k  | ~286 MB | 112 MB | 149 MB |
| 100k | ~372 MB | 212 MB | 281 MB |
| 200k | ~507 MB | 396 MB | 537 MB |

The exact path peaks lower because it never materializes the lazy HNSW layer
during queries.

### Build / update costs (`--ops`, cosine, dim 128)

| Stage | n=10k (+1k inserts, 500 deletes) | n=100k (+5k inserts, 2500 deletes) |
|---|---:|---:|
| build | 1.71 s (0.171 ms/vec) | 3.66 s (0.037 ms/vec) |
| batchInsert | 1.0-1.9 ms/vec | 1.74 ms/vec |
| delete (soft) | 0.9 us/op | 0.2 us/op |
| compact | ~1.0-1.2 s | 3.77 s |
| save (mmap fmt) | 57-65 ms (8.0 MB) | 363 ms (78.2 MB) |
| load (full memory) | 429 ms | 2.25 s |
| loadMmap | 440 ms | 2.06 s |

Recall@10 spot checks against brute force are 1.000 at every lifecycle stage:
before updates, after inserts, after deletes, after compaction, and after both
reload modes. Post-delete/post-reload exactness is guaranteed by fetching
top-(k+deletedCount) flat candidates and filtering soft-deleted rows (see
correctness fixes below).

### Correctness fixes shipped with this verification

1. **Exactness under soft deletions**: `GraphIndex.search`/`batchSearch`
   previously skipped the fused exact path whenever any deletion existed,
   silently degrading to approximate graph traversal after reload. The gate
   now scans top-(k+deletedCount) candidates and filters deleted rows, which
   is provably the true top-k survivors.
2. **Mmap-loaded indexes regain exact search**: Float32-mode mmap storage is
   now flat-scan eligible (its zero-copy buffer already has the row-major
   Float32 layout the kernel reads). Float16/binary/disk-backed staging remain
   excluded.
3. **Deterministic GPU-tier tests**: `FlatGPUSearch.search/batchSearch` accept
   an internal `tierOverride` so regression tests can force the Metal kernel on
   small corpora without mutating shared static state.

### Tests added

`Tests/MetalANNSTests/ExactSearchIntegrationTests.swift`: GPU-tier kernel
exactness across metrics and chunk boundaries, fallback behavior when the
exact path is disabled or over its runtime limit, and a lifecycle test proving
recall stays exactly 1.0 through inserts, deletes, compaction, and both reload
modes. `DiskBackedTests` now asserts recall-vs-brute-force instead of strict
equality between two independently built graph traversals (their NN-descent
builds use GPU atomics and are not bit-deterministic under load).

### Reproduce

```bash
# Scale x metric A/B (exact vs legacy), concurrency=1
for N in 10000 50000 100000 200000; do
  for M in cosine l2 innerproduct; do
    swift run -c release MetalANNSBenchmarks --vector-count $N --query-count 200 \
      --runs 3 --warmup 1 --metric $M
    swift run -c release MetalANNSBenchmarks --vector-count $N --query-count 200 \
      --runs 3 --warmup 1 --metric $M --exact-search-limit 0
  done
done

# Concurrency sweeps
swift run -c release MetalANNSBenchmarks --vector-count 50000 --metric cosine \
  --concurrency-sweep 1,8,16 --runs 3 --warmup 1
swift run -c release MetalANNSBenchmarks --vector-count 50000 --metric cosine \
  --concurrency-sweep 1,8 --runs 1 --exact-search-limit 0

# Lifecycle costs + update-correctness spot checks
swift run -c release MetalANNSBenchmarks --ops
swift run -c release MetalANNSBenchmarks --ops --vector-count 100000 \
  --insert-count 5000 --delete-count 2500

# Deterministic regression suites
swift test --filter ExactSearchIntegrationTests
swift test --filter FlatSearchTests
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
