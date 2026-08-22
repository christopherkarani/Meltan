#include <metal_stdlib>
using namespace metal;

// Fused exact nearest-neighbor distance scan ("flat" brute force).
//
// One thread per (query, row): each thread streams its corpus row
// independently, accumulating dot(q,v) and ||v||^2, then writes the finalized
// distance. No cross-lane reductions, no threadgroup barriers, no sorting.
// Enormous thread counts give the memory system the parallelism it needs to
// saturate bandwidth. Top-K selection happens host-side.
//
// All metrics reduce to dot(q,v) and ||v||^2:
//   cosine       : 1 - qv / (||q|| ||v||)      (||q||^2 precomputed host-side)
//   l2           : ||q||^2 - 2qv + ||v||^2
//   innerProduct : -qv

inline float flat_finalize_metric(
    float dotQV,
    float normVSq,
    uint metricType,
    float queryNormSq
) {
    if (metricType == 1u) {
        return fmax(0.0f, queryNormSq - 2.0f * dotQV + normVSq);
    }
    if (metricType == 2u) {
        return -dotQV;
    }
    float denom = sqrt(queryNormSq) * sqrt(normVSq);
    return denom < 1e-10f ? 1.0f : (1.0f - dotQV / denom);
}

kernel void flat_scan_distances(
    device const float *queries [[buffer(0)]],
    device const float *corpus [[buffer(1)]],
    device const float *queryNorms [[buffer(2)]],
    device float *distances [[buffer(3)]],
    constant uint &queryCount [[buffer(4)]],
    constant uint &vectorCount [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &metricType [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = queryCount * vectorCount;
    if (tid >= total) {
        return;
    }

    uint queryIndex = tid / vectorCount;
    uint rowIndex = tid - queryIndex * vectorCount;

    device const float *queryRow = queries + queryIndex * dim;
    device const float *vectorRow = corpus + rowIndex * dim;
    float queryNormSq = queryNorms[queryIndex];

    float dotQV = 0.0f;
    float normVSq = 0.0f;

    if ((dim & 3u) == 0u) {
        // Aligned float4 path: dim % 4 == 0 keeps every access 16B aligned.
        for (uint d = 0; d < dim; d += 4u) {
            float4 qValue = *((device const float4 *)(queryRow + d));
            float4 vValue = *((device const float4 *)(vectorRow + d));
            dotQV += dot(qValue, vValue);
            normVSq += dot(vValue, vValue);
        }
    } else {
        for (uint d = 0; d < dim; ++d) {
            float qValue = queryRow[d];
            float vValue = vectorRow[d];
            dotQV += qValue * vValue;
            normVSq += vValue * vValue;
        }
    }

    distances[tid] = flat_finalize_metric(dotQV, normVSq, metricType, queryNormSq);
}
