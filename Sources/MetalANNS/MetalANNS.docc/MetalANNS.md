# ``MetalANNS``

On-device vector search for Apple Silicon. Default search is exact.

## Overview

MetalANNS is a Swift vector search library for Metal's unified memory. Default search is a fused exact scan (recall@10 = 1.000). A CAGRA-style NN-Descent graph is still used for construction, mutations, and the large-n fallback. Opt-in `.fast` (IVF-flat) trades a little recall for lower latency.

The API is type-state `VectorIndex`, actor isolation under Swift 6, mutable indexes, mmap persistence, and a metadata filter DSL. Metal compute shaders run on device; simulators and CI use Accelerate (vDSP/BLAS).

### Key Capabilities

- **Exact search by default**, with Metal/NEON kernels and an opt-in IVF `.fast` mode
- **Mutable indexes** — insert, delete, batch update, and compact at runtime
- **Filtered search** with a type-safe result builder DSL supporting boolean logic
- **Multiple persistence modes** — binary, zero-copy mmap, and disk-backed streaming
- **FP16, binary, and product quantization** codecs for memory/speed trade-offs
- **Swift 6 concurrency safe** — `Sendable` enforced at compile time

## Topics

### Essentials

- <doc:GettingStarted>
- ``VectorIndex``
- ``IndexConfiguration``
- ``VectorRecord``

### Search

- ``VectorNeighbor``
- ``QueryFilter``
- ``QueryFilterBuilder``
- ``Field``
- ``Metric``

### Persistence

- ``ReadOnlyLoadMode``
- ``VectorIndexState``

### Advanced Index Types

- ``Advanced``

### Metrics and Diagnostics

- ``IndexMetrics``
- ``MetricsSnapshot``

### Errors

- ``ANNSError``
