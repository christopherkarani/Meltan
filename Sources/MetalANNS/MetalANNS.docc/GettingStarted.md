# Getting Started with MetalANNS

Build a vector index, search it, and persist it to disk.

## Overview

MetalANNS uses a type-state pattern to guide you through the index lifecycle:
**Unbuilt** → **Ready** → **ReadOnly**. The compiler enforces which operations
are available at each stage — you can't search an unbuilt index or mutate a
read-only one.

## Add the Dependency

Add MetalANNS to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/MetalANNS.git", from: "0.2.2")
]
```

Then add it to your target:

```swift
.target(name: "MyApp", dependencies: [
    .product(name: "MetalANNS", package: "MetalANNS")
])
```

## Build an Index

Create a ``VectorIndex`` in the `Unbuilt` state, then call ``VectorIndex/build(records:)``
to construct the graph and transition to `Ready`:

```swift
import MetalANNS

let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)
```

MetalANNS automatically selects a GPU (Metal) or CPU (Accelerate) backend
based on hardware availability.

## Search with Filters

Use the result builder DSL to compose filters at search time:

```swift
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) — score: \(hit.score)")
}
```

## Mutate the Index

A `Ready` index supports live mutations:

```swift
try await ready.insert(VectorRecord(id: "doc_new", vector: newVector))
try await ready.delete(id: "doc_old")
try await ready.compact()
```

## Persist and Load

Save the index to disk and reload it later. Choose the load mode that fits
your access pattern:

```swift
// Save
let url = URL(fileURLWithPath: "/tmp/my-index.mann")
try await ready.save(to: url)

// Full load (mutable)
let loaded = try await VectorIndex<String, VectorIndexState.Unbuilt>.load(from: url)

// Zero-copy mmap (read-only, instant load)
let mmap = try await VectorIndex<String, VectorIndexState.Unbuilt>.loadReadOnly(
    from: url, mode: .mmap
)

// Disk-backed streaming (read-only, low memory)
let disk = try await VectorIndex<String, VectorIndexState.Unbuilt>.loadReadOnly(
    from: url, mode: .diskBacked
)
```

## Power-User Access

For fine-grained control, use the ``Advanced`` namespace to work directly with
low-level index types:

```swift
let graph = GraphIndex(configuration: .default)
try await graph.build(vectors: vectors, ids: ids)
let hits = try await graph.search(query: query, k: 10)
```

Available types:

| Type | Use Case |
|---|---|
| ``GraphIndex`` | Raw CAGRA-style directed graph |
| ``Advanced/StreamingIndex`` | Continuous ingest with background merges |
| ``Advanced/ShardedIndex`` | Large datasets with k-means routing |
| ``Advanced/IVFPQIndex`` | Product quantization for speed/memory trade-offs |
