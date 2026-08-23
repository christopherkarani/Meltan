#ifndef METALANNSC_H
#define METALANNSC_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/// out[r] = dot(query, corpus + r*dim) for r in [0, rowCount).
/// Summation order is fixed by this implementation (NEON 4-lane accumulator,
/// scalar tail), so repeated calls on identical inputs are deterministic.
void mans_f32_dot_rows(const float *corpus,
                       const float *query,
                       int64_t rowCount,
                       int64_t dim,
                       float *out);

/// Exact integer dot products between int8-quantized codes and an int8 query,
/// emitted as Float (exact conversion of the int32 accumulator) so callers
/// can feed SIMD key arithmetic directly.
/// out[r] = sum_d codes[r*dim + d] * queryCodes[d], accumulated in int32.
/// Requires |values| <= 127 and dim small enough that 127*127*dim fits in
/// int32 (dim <= ~133k; callers cap far below).
void mans_i8_dot_rows_f32(const int8_t *codes,
                          const int8_t *queryCodes,
                          int64_t rowCount,
                          int64_t dim,
                          float *out);

/// Exact fp32 dots for a gathered subset of rows (rescore pass).
/// out[i] = dot(query, corpus + indices[i]*dim).
void mans_f32_dot_rows_gather(const float *corpus,
                              const float *query,
                              const uint32_t *indices,
                              int64_t count,
                              int64_t dim,
                              float *out);

#if defined(__cplusplus)
}
#endif

#endif /* METALANNSC_H */
