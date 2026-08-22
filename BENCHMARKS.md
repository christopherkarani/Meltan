# MetalANNS Benchmarks

Last updated: `2026-08-22`

## 2026-08-22 — Hot-path polish, validation matrix, and the exact-search bandwidth wall

Follow-up to the fused exact-search work above. Two host-side costs were
removed from the warm search path, the full {1k…100k} × {128, 384, 768}
validation matrix was run against both in-binary paths, and the remaining gap
to an external "legacy Metal" reference (1.10 ms @10k / 1.72 ms @50k,
dim 384, topK 24) was measured and attributed component by component.

### Changes shipped this round

1. **Corpus-norm cache no longer holds its lock during scans** (`FlatGPUSearch`).
   Norms are published as immutable boxed arrays; readers retain a snapshot
   and run entirely lock-free. Previously every concurrent host-tier search
   serialized on one `NSLock` for the whole selection loop.
2. **Parallel GPU-tier top-K selection.** Batched queries now select rows on
   all cores (`concurrentPerform`, distinct-slot writes through a unique
   buffer pointer); single large scans split into per-core chunk heaps merged
   deterministically by (distance, id) above
   `FlatGPUSearch.parallelSelectionMinVectorCount` (16,384).
3. **Harness additions**: `--dimension <n>` flag and a `--probe` mode that
   decomposes flat-search cost components against a raw `VectorBuffer`.

### Correctness regressions caught by new tests (and fixed)

Chunked selection initially emitted **chunk-local ids**: candidate `id =
UInt32(index)` had to become `idBase + index` in *both* the heap-fill *and*
heap-replacement branches — missing the replacement branch silently returned
wrong rows (recall@10 fell to 0.356 at 50k). Two regression tests now pin
this: `parallelTierSelectionStaysExact` (corpus past the fan-out threshold,
tier forced, ties straddling lane boundaries) and
`concurrentHostSearchesSurviveCacheChurn` (TaskGroup storm over a shared
buffer with interleaved cache invalidation).

### Validation summary

- Full Swift Testing suite: **264 tests / 64 suites pass** (run twice: after
  code changes and after lint formatting).
- `swift-format lint --strict` clean.
- Recall@k = **1.000** on the exact path at every size/dim/metric tested,
  including after inserts, soft deletes, compaction, save/load, and mmap
  reloads (`--ops` spot checks).
- Sanitizers: **unavailable on this OS/toolchain** — macOS 26 refuses to load
  both TSan and ASan runtime dylibs ("Sanitizer load violates platform
  policy"). Swift 6 strict concurrency plus the concurrency stress suites
  provide the available race coverage.

### Warm latency, interleaved A/B (dim 384, k=24, cosine, 200 queries ×3 runs +1 warmup, seed 42)

Legacy column is the same binary's graph-traversal path
(`--exact-search-limit 0`), alternating L/E/E/L windows to cancel machine
drift (Apple GPU clocks swing p50 ±40% between windows on this machine; A/B
claims below come from same-window pairs).

| Corpus | Legacy p50 | Exact p50 | vs in-binary legacy | vs external bar¹ |
|---|---|---|---|---|
| 10k    | 3.29–3.34 ms | 0.095–0.10 ms | **33×**  | **11× ✓ target met** |
| 50k    | 16.55–17.38 ms | 0.44–0.46 ms | **37×** | **3.8× ✗ wall** |

¹ External reference: user-supplied legacy Metal baseline (1.10 ms @10k /
1.72 ms @50k). At 50k the exact path cannot reach 10× — see bandwidth wall.

### Full matrix (single window, favorable clock state; treat as relative shape)

Exact-path p50 ms (recall 1.000 everywhere):

| Vectors | dim 128 | dim 384 | dim 768 |
|---|---|---|---|
| 1k     | 0.04 | 0.04 | 0.04 |
| 5k     | 0.06 | 0.06 | 0.06 |
| 10k    | 0.10 | 0.09 | 0.09 |
| 50k    | 0.35 | 0.33 | 0.35 |
| 100k   | 0.50 | 0.51 | 0.50 |

Same-window legacy p50: 0.32 / 1.53 / 2.99 / 15.4 / 31.8 ms; legacy recall@10
degrades to 0.989 at 100k while the exact path holds 1.000.

Build time is unchanged by this round (~0.22 ms/vec small-n, 1.6 s @10k,
4.4 s @100k); index-resident memory unchanged (~287 MB @50k, peak RSS 114 MB;
~371 MB @100k×768, peak 209 MB).

Lifecycle (`--ops`, dim 128): build 1.61 s @10k / 4.34 s @100k; batchInsert
1.3–2.0 ms/vec; delete ~0.3 µs/op; compact 0.9 s / 4.8 s; save 70 ms (8 MB) /
487 ms (78 MB); load(fullMemory) 175 ms / 3.44 s; loadMmap 162 ms / 3.28 s.
Recall 1.000 before/after inserts/deletes/compaction and after both reload
modes at both scales.

Throughput under concurrency (50k×128 cosine): exact 1,610 QPS @c=1 →
**6,795 QPS @c=8** → 6,253 @c=16 (recall 1.000 throughout); legacy 58 →
62 QPS. Steady-state advantage ≈ **110× at c=8**.

### Why 10× at 50k×384 is unreachable for exact search (component evidence)

Steady-state decomposition of a single warm query, measured via `--probe`,
standalone dispatch micro-benchmarks, and kernel variants:

| Component @50k×384 | Measured |
|---|---|
| Empty dispatch+sync round trip | ~150–180 µs |
| `flat_scan_distances` streaming 73.2 MB | ~240–360 µs (155–305 GB/s effective, cache-state dependent) |
| Host top-K heap over 50k distances | 30 µs serial → parallelized this round |
| Actor/id-mapping glue | ~15–25 µs |

- DRAM floor: 73.2 MB ÷ ~300 GB/s (M3 Max peak) ≈ **245 µs**, before any
  dispatch tax. The 172 µs target (= 1.72 ms ÷ 10) sits **below the physical
  floor** for reading the corpus once per query in Float32.
- Kernel-shape study: unrolled-float4 and simdgroup-reduction variants landed
  within noise of the production kernel — no shape win available.
- Host BLAS loses above ~16k vectors (sgemv streams at 66–119 GB/s vs GPU
  155–305 GB/s); the ≤32,768-vector host-tier crossover remains correct.
- Batching amortizes well (64-query dispatch → ~380 µs/query at 50k×384) and
  is the recommended throughput mode (`batchSearch`).

Maximum measured improvement vs the external baseline is therefore **11× @10k**
(target met) and **3.8–5.2× @50k across clock windows** (bandwidth-walled).
Reaching sub-floor latency at 50k+ would require reducing bytes per query —
Float16 mirrors or quantized pruning — which changes accuracy guarantees and
was explicitly out of scope. Future options if ever wanted: persistent-kernel
query mailbox (removes most of the dispatch tax) and opt-in f16 scan tier.

### Reproduce

```bash
swift build -c release

# Interleaved A/B at the reference protocol (dim 384, k24)
.build/release/MetalANNSBenchmarks --vector-count 10000 --dimension 384 --k 24 \
  --query-count 200 --runs 3 --warmup 1 --seed 42
.build/release/MetalANNSBenchmarks --vector-count 50000 --dimension 384 --k 24 \
  --query-count 200 --runs 3 --warmup 1 --seed 42
# ...repeat each with --exact-search-limit 0 for the legacy column

# Full matrix + lifecycle + concurrency
for N in 1000 5000 10000 50000 100000; do for D in 128 384 768; do
  .build/release/MetalANNSBenchmarks --vector-count $N --dimension $D --k 24 \
    --query-count 200 --runs 3 --warmup 1 --seed 42 --csv-out m_${N}_${D}.csv
done; done
.build/release/MetalANNSBenchmarks --ops --vector-count 100000 --insert-count 5000 --delete-count 2500
.build/release/MetalANNSBenchmarks --vector-count 50000 --concurrency-sweep 1,8,16 --k 24

# Component attribution
.build/release/MetalANNSBenchmarks --probe

# Regression suites for this round
swift test --filter ExactSearchIntegrationTests
swift test --filter FlatSearchTests
```


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
