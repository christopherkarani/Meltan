# ``VectorIndex``

A type-safe, stateful vector index with compiler-enforced lifecycle constraints.

## Overview

`VectorIndex` uses Swift's type system to prevent invalid operations at compile time:
- You can only call ``VectorIndex/build(records:)`` on an `Unbuilt` index
- You can only call ``VectorIndex/search(query:topK:metric:filter:)`` on a `Ready` or `ReadOnly` index
- The ``VectorIndex/advanced`` escape hatch exposes the underlying `_GraphIndex` actor for power users

## Topics

### Creating an Index

- ``VectorIndex/init(configuration:)``
- ``VectorIndex/build(records:)``
- ``VectorIndex/build(vectors:ids:)``

### Searching

- ``VectorIndex/search(query:topK:metric:filter:)``
- ``VectorIndex/batchSearch(queries:topK:metric:filter:)``
- ``VectorIndex/rangeSearch(query:maxDistance:limit:metric:filter:)``

### Mutation

- ``VectorIndex/insert(_:)``
- ``VectorIndex/batchInsert(_:)``
- ``VectorIndex/delete(id:)``
- ``VectorIndex/compact()``

### Persistence

- ``VectorIndex/save(to:)``
- ``VectorIndex/load(from:)``
- ``VectorIndex/loadReadOnly(from:mode:)``
