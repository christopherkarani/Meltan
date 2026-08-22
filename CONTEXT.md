# MetalANNS

An approximate-nearest-neighbour index for Apple Silicon: vectors are organized in
proximity graphs, searched on GPU or CPU, persisted, sharded, and streamed.

## Language

### Search

**vector search**:
Finding the nearest stored vectors for a query vector; the operation every search
adapter implements.
_Avoid_: ANN query, similarity query

**graph search**:
Beam search over the proximity graph. One of several adapters behind a vector search.
_Avoid_: HNSW search

**exact search**:
Vector search by brute-force distance scan over the whole corpus — no graph;
recall 1.0 by construction.
_Avoid_: flat scan, brute force

**search path**:
Which adapter runs a vector search — `auto` (heuristic choice, falls back freely),
or `exact`, `gpu`, `cpu` (strict: throws if the requested adapter cannot run).
_Avoid_: forceGPU, execution mode, backend flag

**candidate**:
An `(internalID, score)` pair produced by vector search, before deletion filtering
and ID mapping.
_Avoid_: result, hit

**neighbor**:
A candidate after filtering and ID resolution, carrying an external key — what a
caller receives from `search`.
_Avoid_: match, record
