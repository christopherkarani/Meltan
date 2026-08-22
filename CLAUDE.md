# CLAUDE.md

Guidance for AI assistants working in the MetalANNS repository.

## Project Overview

**MetalANNS** is a GPU-native approximate nearest neighbor search library for Apple Silicon. It uses CAGRA-style NN-Descent graphs (fully GPU-parallelizable, in contrast to sequential HNSW) and dispatches work through Metal compute shaders, with an Accelerate/vDSP fallback for simulators and non-Metal environments.

- Language: Swift 6 (strict concurrency, `swiftLanguageMode(.v6)` on all targets)
- Platforms: macOS 14+, iOS 17+, visionOS 1+
- Distribution: Swift Package Manager library (`Package.swift` at repo root)
- GPU: Metal compute shaders bundled as a target resource
- External dependency: `GRDB.swift` (SQLite) for structured metadata + persistence (pulled via SPM)

## Build / Test / Benchmark Commands

```bash
# Build the package (debug)
swift build

# Build release (required for realistic benchmark numbers)
swift build -c release

# Run the full test suite (Swift Testing, not XCTest)
swift test

# Run a filtered subset (Swift Testing filter matches suite/test names)
swift test --filter MetalDeviceTests
swift test --filter FullGPUSearchTests
swift test --filter IVFPQComprehensiveTests.benchmarkSearchThroughput

# Run the benchmark CLI (synthetic dataset by default)
swift run -c release MetalANNSBenchmarks --query-count 200 --runs 3 --warmup 1

# Compare GraphIndex vs IVFPQ
swift run -c release MetalANNSBenchmarks --ivfpq --query-count 200 --runs 3 --warmup 1

# Benchmark flags of note: --dataset <path.annbin>, --sweep, --sweep-efsearch,
# --csv-out, --json-out, --metric, --degree, --efsearch, --k, --ivfpq-*
```

The benchmark CLI lives in `Sources/MetalANNSBenchmarks/main.swift`. See `BENCHMARKS.md` for current numbers and reproduction instructions.

## Repository Layout

```
Sources/
  MetalANNSCore/          # Low-level building blocks (Metal, kernels, buffers, algorithms)
    Shaders/              # .metal compute shaders (bundled via resources: [.process("Shaders")])
  MetalANNS/              # Public API surface built on top of Core
    Storage/              # GRDB/SQLite-backed persistence (IndexDatabase, StreamingDatabase)
    MetalANNS.docc/       # DocC catalog (MetalANNS.md, GettingStarted.md)
  MetalANNSBenchmarks/    # Executable benchmark runner (main.swift, BenchmarkRunner, etc.)
Tests/
  MetalANNSTests/         # Swift Testing suites covering all public + internal APIs
docs/                     # Design docs, plans, prompts, audit reports (gitignored)
tasks/                    # Phase todos, review protocol, lessons learned (gitignored)
scripts/                  # Utility scripts, e.g. convert_hdf5.py (gitignored)
locales/                  # Translated READMEs
Package.swift             # SPM manifest
BENCHMARKS.md             # Current measured performance
README.md                 # Marketing + API overview
```

Note: `docs/`, `tasks/`, and `scripts/` are listed in `.gitignore` (project metadata excluded from the shipped package), so do not assume new files there will be committed automatically.

### Target Graph

- `MetalANNSCore` — no dependencies, ships Metal shaders as resources
- `MetalANNS` — depends on `MetalANNSCore` + `GRDB`, exposes the public API
- `MetalANNSBenchmarks` — executable, depends on `MetalANNS` + `MetalANNSCore`
- `MetalANNSTests` — depends on all three targets + `GRDB`

## Architecture at a Glance

### Compute Backends (`Sources/MetalANNSCore`)

- `ComputeBackend` protocol with `MetalBackend` (GPU) and `AccelerateBackend` (CPU vDSP) implementations
- `BackendFactory.makeBackend()` chooses Metal when available; simulators always use Accelerate
- `MetalContext` owns the `MTLDevice`, `MTLCommandQueue`, a `CommandQueuePool` for pipelined concurrent work, the `PipelineCache` actor, and the `SearchBufferPool`
- `MetalContext.loadLibrary` has a three-tier fallback: `Bundle.module` default library → device default library → compile bundled `.metal` sources at runtime. Preserve this behavior — it keeps benchmarks and tests working on machines where the SPM resource pipeline does not ship a prebuilt `.metallib`.

### Storage and Buffers

- `VectorStorage` protocol abstracts Float32 (`VectorBuffer`), Float16 (`Float16VectorBuffer`), binary (`BinaryVectorBuffer`), PQ codes (`PQVectorBuffer`), and mmap/disk-backed variants (`DiskBackedVectorBuffer`)
- `GraphBuffer` holds the fixed-degree directed adjacency list used by CAGRA-style search
- `MetadataBuffer` / `MetadataStore` / `IDMap` handle metadata + external↔internal ID mapping

### Index Construction and Search

- `NNDescentCPU` / `NNDescentGPU` build the initial directed graph (GPU kernel requires `degree` to be a power of two ≤ 64; see `IndexConfiguration.validateGPUConstructionConstraints`)
- `GraphPruner`, `GraphRepairer` maintain graph quality after edits
- `BeamSearchCPU`, `HNSWSearchCPU`, `SearchGPU`, `FullGPUSearch`, `GPUADCSearch` provide alternative query paths; `_GraphIndex` picks between them based on workload size (small workloads run on CPU because GPU submission overhead dominates — see thresholds in `ANNSIndex.swift`)
- `HNSWLayers` / `HNSWBuilder` layer skip-lists above the base graph for CPU search
- `IVFPQIndex` uses `ProductQuantizer` + `KMeans` coarse centroids for quantized search
- `IndexSerializer` / `MmapIndexLoader` handle persistence, including zero-copy mmap and disk-backed read-only modes

### Public API (`Sources/MetalANNS`)

The public API is intentionally narrow. Most work goes through `VectorIndex<Key, State>`:

- **Type-state machine**: `VectorIndexState.Unbuilt`, `.Ready`, `.ReadOnly`. The compiler enforces that `.search` only exists on `.Ready`/`.ReadOnly`, mutations only on `.Ready`, etc. Preserve this separation when adding API.
- **Generic key**: `IndexKey` (conformed by `String` and `UInt64`). Internally everything is stringified for compatibility with `_GraphIndex`.
- `IndexConfiguration` holds `degree`, `metric`, `efConstruction`, `efSearch`, `maxIterations`, `useFloat16`, `useBinary`, `convergenceThreshold`, and embedded `HNSWConfiguration` + `RepairConfiguration`.
- `QueryFilter` + `QueryFilterBuilder` give a result-builder DSL on top of the internal `_LegacySearchFilter` (and/or/not, equals, range, set-membership).
- `Metric`, `VectorRecord`, `VectorNeighbor`, `SearchResult`, `ANNSError` round out the surface.

The lower-level actors (`_GraphIndex`, `_StreamingIndex`, `_ShardedIndex`, `_IVFPQIndex`) are prefixed with `_` to mark them as power-user APIs, and are re-exported through the `Advanced` namespace (`Advanced.GraphIndex`, `Advanced.StreamingIndex`, `Advanced.ShardedIndex`, `Advanced.IVFPQIndex`). Prefer extending `VectorIndex` for user-facing features and only drop into `Advanced` when you need raw control.

- `_GraphIndex` (`Sources/MetalANNS/ANNSIndex.swift`, ~1400 lines) — the core mutable CAGRA index used by `VectorIndex`
- `_StreamingIndex` (`StreamingIndex.swift`) — two-level delta/base merge for continuous ingest
- `_ShardedIndex` (`ShardedIndex.swift`) — k-means routed shards for large datasets
- `_IVFPQIndex` (`IVFPQIndex.swift`) — IVF + product quantization

### Persistence

- Custom binary format written by `IndexSerializer`, loadable in three modes: full-memory, zero-copy `mmap`, and `diskBacked` streaming (`ReadOnlyLoadMode`).
- SQLite (GRDB) is used for structured sidecar data under `Sources/MetalANNS/Storage/`:
  - `IndexDatabase` — idmap, config, soft_deletion, numeric idmap; WAL journal mode; uses `DatabaseMigrator` with named migrations (`v1-foundation`, `v2-idmap-numeric`, …). Always add new schema via a new named migration rather than editing existing ones.
  - `StreamingDatabase` — persistence for `_StreamingIndex`
  - `SQLiteStructuredStore` — generic metadata column store
- `MmapIndexLoader` is tightly coupled to the binary layout; changes to on-disk layout require a version bump and compatibility handling in `IndexDatabase` + loader fallbacks (see `IDMapPersistenceCompat.swift`).

## Conventions for Contributors (and AI Agents)

### Swift 6 Concurrency

- The project builds under strict concurrency. All three library targets set `swiftSettings: [.swiftLanguageMode(.v6)]`. Do not downgrade.
- Public stateful types are `actor`s (`_GraphIndex`, `_StreamingIndex`, `_ShardedIndex`, `_IVFPQIndex`, `PipelineCache`). Keep new mutable components as actors or genuinely immutable `Sendable` structs.
- `@unchecked Sendable` is reserved for Metal wrappers (`MetalContext`, `IndexDatabase`) where ownership is enforced by external contracts. Do not sprinkle it elsewhere to silence warnings — fix the underlying data race.
- Prefer `Sendable` value types in the public API. `VectorRecord`, `VectorNeighbor`, `IndexConfiguration`, `QueryFilter`, `Metric`, etc. are all `Sendable`.

### Testing

- Tests use **Swift Testing** (`import Testing`, `@Suite`, `@Test`, `#expect`), not XCTest. Do not add `import XCTest` — see `tasks/review-protocol.md`.
- Use `@testable import MetalANNSCore` / `@testable import MetalANNS` when you need internal symbols.
- Use `SeededGenerator` from `Tests/MetalANNSTests/TestUtilities.swift` for reproducible random data.
- GPU-specific suites (`MetalDeviceTests`, `MetalSearchTests`, `FullGPUSearchTests`, `IVFPQGPUTests`, `NNDescentGPUTests`, `MetalDistanceTests`, `MetalContextMultiQueueTests`) need a functioning Metal device — they will skip or fail on machines without Metal. Always run at least `MetalDeviceTests` when touching GPU code.
- Parity tests (`GPUCPUParityTests`) guard that CPU and GPU search paths agree; add a parity test any time you introduce a new GPU kernel with a CPU equivalent.
- Per `tasks/lessons.md`, add kernel-level deterministic regression tests for GPU correctness invariants (don't rely only on end-to-end recall).

### Metal Shaders (`Sources/MetalANNSCore/Shaders/*.metal`)

- Shaders are bundled as processed resources via `resources: [.process("Shaders")]`. When adding a new `.metal` file, no additional Package.swift edits are needed — just drop it in the `Shaders/` directory.
- Kernels run under the `com.metalanns` logging subsystem (see `os.log` loggers in the Swift counterparts). Keep buffer-index layouts in Swift encoders in sync with `[[buffer(N)]]` declarations.
- Common pitfalls called out in the review protocol and lessons file:
  - Guard `if (tid >= n) return` at kernel entry.
  - Preserve pairwise symmetry in NN-Descent `local_join` (update both `a` and `b`).
  - Recompute worst-distance bounds per candidate pair inside CAS loops — do not cache across iterations.
  - Use `UINT_MAX` / `memory_order_relaxed` (Metal dialect), not their lowercase aliases.

### Coding Style

- `swift-tools-version: 6.0`, uses language mode 6. No workarounds for older tool versions.
- Modules log via `os.Logger` with subsystem `com.metalanns` and a per-file category.
- Errors flow through `ANNSError` (`Sources/MetalANNSCore/Errors.swift`). Prefer extending that enum over introducing a new error type for core functionality.
- Existing files are long and cohesive (e.g. `ANNSIndex.swift` is ~1400 lines). That's intentional — don't split them up speculatively. Match the file's existing structure when adding methods.
- Keep the public API small. If a feature only needs internal visibility, don't mark it `public`.

### Git Workflow

- Trunk branch is `main`. Development for this session should happen on `claude/add-claude-documentation-ZZaVC` per the session instructions.
- Commit message style from `git log`: Conventional-ish prefixes — `feat:`, `fix:`, `perf:`, `docs:`, `chore:`, `refactor:`. Sentence-case descriptions. Keep bodies focused on *why*.
- Historical commits show one-logical-change-per-commit; the perf branch history in particular uses multiple small commits rather than a single omnibus change.
- `docs/`, `tasks/`, and `scripts/` are in `.gitignore`. They exist in the working tree as project workbooks but are intentionally excluded from the package — never move code that must ship into those directories.

### Performance Guard-Rails

`BENCHMARKS.md` records the current baseline. When changing hot paths, re-run the relevant benchmark/test and update `BENCHMARKS.md` if numbers shift materially. Key things the current baseline depends on (do not regress silently):

- `_GraphIndex` routes small builds/searches to CPU — see `minGPUConstructionNodeCount` at the top of `ANNSIndex.swift` (construction) and the workload gates in `VectorSearch.swift` (`MetalANNSCore`). Removing this workload-aware gating historically caused a >65× regression on small workloads.
- GPU search and GPU ADC reuse workspaces from `SearchBufferPool`; do not allocate fresh `MTLBuffer`s per query/expansion.
- IVFPQ add/train/search path avoids per-call sorting and planning overhead; watch for accidental reintroduction when refactoring.
- `KMeans` Lloyd iterations parallelize safely — keep the subspace-materialization fast paths intact.

## When Making Changes

1. **Read before you write.** Files like `ANNSIndex.swift`, `IVFPQIndex.swift`, `StreamingIndex.swift`, and `MmapIndexLoader.swift` encode subtle invariants — read the relevant section fully before editing.
2. **Match the existing layer.** New public features go through `VectorIndex` in `Sources/MetalANNS`; new algorithmic primitives belong in `Sources/MetalANNSCore`; new `.metal` code goes in `Sources/MetalANNSCore/Shaders`.
3. **Cover parity when touching GPU paths.** Run `swift test --filter GPUCPUParityTests` plus the relevant `Metal*Tests` suite.
4. **Preserve type-state safety.** Do not add mutating operations to `VectorIndexState.ReadOnly` or searches to `.Unbuilt`.
5. **Add a schema migration, never edit one.** `IndexDatabase.migrate` uses named migrations; changing an existing migration breaks already-persisted indexes.
6. **Update `BENCHMARKS.md`** if you change observed numbers; mention the reproduction command.
7. **Check `tasks/lessons.md`** for accumulated gotchas before modifying NN-Descent, local_join, graph search, or serialization fallbacks.
