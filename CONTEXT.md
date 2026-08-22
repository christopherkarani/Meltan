# MetalANNS

An approximate-nearest-neighbour index for Apple Silicon: vectors are organized in
proximity graphs, searched on GPU or CPU, persisted, sharded, and streamed.

## Language

### Search

**graph search**:
Beam search over the proximity graph; the operation every search backend implements.
_Avoid_: vector scan, ANN query

**search path**:
Which adapter runs a graph search — `auto` (heuristic choice, falls back freely),
`gpu`, or `cpu` (strict: throws if the requested adapter cannot run).
_Avoid_: forceGPU, execution mode, backend flag

**candidate**:
An `(internalID, score)` pair produced by graph search, before deletion filtering
and ID mapping.
_Avoid_: result, hit

**neighbor**:
A candidate after filtering and ID resolution, carrying an external key — what a
caller receives from `search`.
_Avoid_: match, record
