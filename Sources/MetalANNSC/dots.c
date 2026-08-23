#include "MetalANNSC.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define MANS_NEON 1
#endif

void mans_f32_dot_rows(const float *corpus,
                       const float *query,
                       int64_t rowCount,
                       int64_t dim,
                       float *out) {
    int64_t r = 0;
#if defined(MANS_NEON)
    // Vectorized path: 16B-aligned corpus and dim divisible by 4.
    if (dim >= 16 && (dim & 3) == 0 && (((uintptr_t)corpus & 15u) == 0u)) {
        for (; r < rowCount; ++r) {
            const float *row = corpus + r * dim;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int64_t d = 0;
            for (; d + 16 <= dim; d += 16) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(row + d),      vld1q_f32(query + d));
                acc1 = vfmaq_f32(acc1, vld1q_f32(row + d + 4),  vld1q_f32(query + d + 4));
                acc2 = vfmaq_f32(acc2, vld1q_f32(row + d + 8),  vld1q_f32(query + d + 8));
                acc3 = vfmaq_f32(acc3, vld1q_f32(row + d + 12), vld1q_f32(query + d + 12));
            }
            for (; d < dim; d += 4) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(row + d), vld1q_f32(query + d));
            }
            acc0 = vaddq_f32(acc0, vaddq_f32(acc1, vaddq_f32(acc2, acc3)));
            float32x2_t lo = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
            out[r] = vget_lane_f32(vpadd_f32(lo, lo), 0);
        }
        return;
    }
#endif
    for (; r < rowCount; ++r) {
        const float *row = corpus + r * dim;
        float acc = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            acc += row[d] * query[d];
        }
        out[r] = acc;
    }
}

void mans_i8_dot_rows_f32(const int8_t *codes,
                          const int8_t *queryCodes,
                          int64_t rowCount,
                          int64_t dim,
                          float *out) {
    // Exact int accumulation, single conversion at the end.
    for (int64_t r = 0; r < rowCount; ++r) {
        const int8_t *row = codes + r * dim;
        int64_t d = 0;
        int32_t total = 0;
#if defined(MANS_NEON)
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (; d + 32 <= dim; d += 32) {
            int8x16_t c0 = vld1q_s8(row + d);
            int8x16_t c1 = vld1q_s8(row + d + 16);
            int8x16_t q0 = vld1q_s8(queryCodes + d);
            int8x16_t q1 = vld1q_s8(queryCodes + d + 16);
            acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(c0), vget_low_s8(q0)));
            acc1 = vpadalq_s16(acc1, vmull_s8(vget_high_s8(c0), vget_high_s8(q0)));
            acc2 = vpadalq_s16(acc2, vmull_s8(vget_low_s8(c1), vget_low_s8(q1)));
            acc3 = vpadalq_s16(acc3, vmull_s8(vget_high_s8(c1), vget_high_s8(q1)));
        }
        for (; d + 16 <= dim; d += 16) {
            int8x16_t c = vld1q_s8(row + d);
            int8x16_t q = vld1q_s8(queryCodes + d);
            acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(c), vget_low_s8(q)));
            acc1 = vpadalq_s16(acc1, vmull_s8(vget_high_s8(c), vget_high_s8(q)));
        }
        total = vaddvq_s32(vaddq_s32(acc0, vaddq_s32(acc1, vaddq_s32(acc2, acc3))));
#endif
        for (; d < dim; ++d) {
            total += (int32_t)row[d] * (int32_t)queryCodes[d];
        }
        out[r] = (float)total;
    }
}

void mans_f32_dot_rows_gather(const float *corpus,
                              const float *query,
                              const uint32_t *indices,
                              int64_t count,
                              int64_t dim,
                              float *out) {
#if defined(MANS_NEON)
    if ((dim & 3) == 0) {
        for (int64_t i = 0; i < count; ++i) {
            const float *row = corpus + (int64_t)indices[i] * dim;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            int64_t d = 0;
            for (; d + 8 <= dim; d += 8) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(row + d),     vld1q_f32(query + d));
                acc1 = vfmaq_f32(acc1, vld1q_f32(row + d + 4), vld1q_f32(query + d + 4));
            }
            for (; d < dim; d += 4) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(row + d), vld1q_f32(query + d));
            }
            acc0 = vaddq_f32(acc0, acc1);
            float32x2_t lo = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
            out[i] = vget_lane_f32(vpadd_f32(lo, lo), 0);
        }
        return;
    }
#endif
    for (int64_t i = 0; i < count; ++i) {
        const float *row = corpus + (int64_t)indices[i] * dim;
        float acc = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            acc += row[d] * query[d];
        }
        out[i] = acc;
    }
}
