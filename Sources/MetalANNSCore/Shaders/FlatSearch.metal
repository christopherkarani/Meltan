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

// Bound-only scan for the exact residual cascade. The PCA projection planes
// are precomputed once on the host, so each query reads width projections per
// row rather than all dim raw vector coordinates. The host then exact-rescores
// only rows whose bound can beat the seed cutoff.
struct residual_bound_parameters {
    uint vectorCount;
    uint headWidth;
    uint metricType;
    float queryTailNorm;
    float queryNorm;
    float queryNormSq;
    float dotMean;
    float meanNormSq;
    float slack;
    float absoluteSlack;
};

kernel void residual_compute_bounds(
    device const float *projection [[buffer(0)]],
    device const float *vDotMu [[buffer(1)]],
    device const float *rowNormSq [[buffer(2)]],
    device const float *tailNorm [[buffer(3)]],
    device const float *queryHead [[buffer(4)]],
    device float *lowerBounds [[buffer(5)]],
    constant residual_bound_parameters &params [[buffer(6)]],
    uint rowIndex [[thread_position_in_grid]]
) {
    if (rowIndex >= params.vectorCount) {
        return;
    }

    float headDot = 0.0f;
    // Projection is column-major in the GPU cache. A SIMD group therefore
    // reads adjacent rows for each component instead of striding by width.
    for (uint column = 0; column < params.headWidth; ++column) {
        headDot += queryHead[column]
            * projection[column * params.vectorCount + rowIndex];
    }

    float tailUpper = params.queryTailNorm * tailNorm[rowIndex]
        * (1.0f + params.slack) + params.absoluteSlack;
    float meanTerm = params.dotMean + vDotMu[rowIndex];
    float inflation = params.slack * (
        fabs(headDot) + fabs(meanTerm) + fabs(params.meanNormSq)
        + fabs(tailUpper)
    );
    float dotUpper = headDot * (1.0f + params.slack) + tailUpper
        + meanTerm - params.meanNormSq + inflation + params.absoluteSlack;
    float normSquared = rowNormSq[rowIndex];

    if (params.metricType == 0u) {
        if (normSquared < 1e-20f || params.queryNorm < 1e-10f) {
            lowerBounds[rowIndex] = 1.0f;
            return;
        }
        float denominator = params.queryNorm * sqrt(normSquared);
        float similarityUpper = (dotUpper - params.absoluteSlack) / denominator;
        if (isnan(similarityUpper) || similarityUpper > 1.0f) {
            similarityUpper = 1.0f;
        }
        lowerBounds[rowIndex] = 1.0f - similarityUpper;
    } else if (params.metricType == 2u) {
        lowerBounds[rowIndex] = -dotUpper;
    } else {
        float lowerNormSquared = normSquared * (1.0f - params.slack)
            - params.absoluteSlack;
        if (lowerNormSquared < 0.0f) {
            lowerNormSquared = 0.0f;
        }
        float gap = params.queryNormSq - 2.0f * dotUpper + lowerNormSquared;
        lowerBounds[rowIndex] = max(gap, 0.0f);
    }
}
