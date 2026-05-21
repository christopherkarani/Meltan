# ``IndexConfiguration``

Controls the trade-offs between build speed, search speed, and recall quality.

## Overview

`IndexConfiguration` defines the core parameters for graph construction and search:

- ``IndexConfiguration/degree`` — the maximum out-degree of each node in the CAGRA graph
- ``IndexConfiguration/efSearch`` — the beam width during search (larger = better recall, slower)
- ``IndexConfiguration/metric`` — the distance metric (cosine, L2, innerProduct, hamming)

## GPU Construction Constraints

The Metal NN-Descent kernel requires:
- `degree` must be a power of two
- `degree` must be ≤ 64

When these constraints are violated, the library automatically falls back to CPU construction.

## Topics

### Core Parameters

- ``IndexConfiguration/degree``
- ``IndexConfiguration/efSearch``
- ``IndexConfiguration/metric``

### Advanced Parameters

- ``IndexConfiguration/useFloat16``
- ``IndexConfiguration/useBinary``
- ``IndexConfiguration/maxIterations``
- ``IndexConfiguration/convergenceThreshold``
- ``IndexConfiguration/hnswConfiguration``
- ``IndexConfiguration/repairConfiguration``
