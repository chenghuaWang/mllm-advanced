/**
 * @file hgemm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/hgemm.hpp"
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
#include <algorithm>
#include <arm_neon.h>

// Include micro-kernel variants
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && __ARM_ARCH >= 8
#define USE_FP16_FMA 1
#else
#warning "FP16 FMA not enabled, fallback to FP32 accumulation"
#endif

namespace mllm::arm {

static inline void _hgemm_mk_nk_mn_tile_s8_k16_V2(const float16_t* __restrict A,
                                                  const float16_t* __restrict B,
                                                  const float16_t* __restrict BIAS,
                                                  float16_t* __restrict C, size_t ACTUAL_TILE_M,
                                                  size_t ACTUAL_TILE_N, size_t M, size_t K,
                                                  size_t N) {
  constexpr size_t S_TILE = 8;
  constexpr size_t K_TILE = 16;

  const size_t M_TILE_SIZE = std::min(S_TILE, ACTUAL_TILE_M);
  const size_t N_TILE_SIZE = std::min(S_TILE, ACTUAL_TILE_N);

#ifdef USE_FP16_FMA
  float16x8_t c[8] = {vdupq_n_f16(0)};
#else
  float32x4_t c[8][2] = {{vdupq_n_f32(0)}};
#endif

  if (BIAS) {
    for (size_t i = 0; i < M_TILE_SIZE; ++i) {
#ifdef USE_FP16_FMA
      float16x8_t bias = vld1q_f16(BIAS);
      for (size_t j = 0; j < N_TILE_SIZE; j += 8) { c[i] = vaddq_f16(c[i], bias); }
#else
      float32x4_t bias_low = vcvt_f32_f16(vld1_f16(BIAS));
      float32x4_t bias_high = vcvt_f32_f16(vld1_f16(BIAS + 4));
      c[i][0] = vaddq_f32(c[i][0], bias_low);
      c[i][1] = vaddq_f32(c[i][1], bias_high);
#endif
    }
  }

  for (size_t k = 0; k < K; k += K_TILE) {
    const size_t k_end = std::min(k + K_TILE, K);
    for (size_t k_step = k; k_step < k_end; k_step += 8) {
      float16x8_t a[S_TILE];
      for (size_t i = 0; i < M_TILE_SIZE; ++i) { a[i] = vld1q_f16(A + i * K + k_step); }

      float16x8_t b[S_TILE];
      for (size_t j = 0; j < N_TILE_SIZE; ++j) { b[j] = vld1q_f16(B + j * K + k_step); }

      for (size_t i = 0; i < M_TILE_SIZE; ++i) {
        const float16x8_t ai = a[i];
        for (size_t j = 0; j < N_TILE_SIZE; j += 4) {
#ifdef USE_FP16_FMA
          c[i] = vfmaq_f16(c[i], ai, b[j]);
          c[i] = vfmaq_f16(c[i], ai, b[j + 1]);
          c[i] = vfmaq_f16(c[i], ai, b[j + 2]);
          c[i] = vfmaq_f16(c[i], ai, b[j + 3]);
#else
          float32x4_t ai_low = vcvt_f32_f16(vget_low_f16(ai));
          float32x4_t ai_high = vcvt_f32_f16(vget_high_f16(ai));

          for (size_t n = 0; n < 4; ++n) {
            float32x4_t bj_low = vcvt_f32_f16(vget_low_f16(b[j + n]));
            float32x4_t bj_high = vcvt_f32_f16(vget_high_f16(b[j + n]));

            c[i][0] = vmlaq_lane_f32(c[i][0], bj_low, ai_low, n);
            c[i][1] = vmlaq_lane_f32(c[i][1], bj_high, ai_high, n);
          }
#endif
        }
      }
    }
  }

  for (size_t i = 0; i < M_TILE_SIZE; ++i) {
#ifdef USE_FP16_FMA
    vst1q_f16(C + i * N, c[i]);
#else
    float16x8_t result = vcombine_f16(vcvt_f16_f32(c[i][0]), vcvt_f16_f32(c[i][1]));
    vst1q_f16(C + i * N, result);
#endif
  }
}

void hgemm_mk_nk_mn_V1(const float16_t* __restrict lhs, const float16_t* __restrict rhs,
                       float16_t* __restrict dst, size_t M, size_t K, size_t N,
                       const float16_t* __restrict bias, int threads) {
  constexpr size_t TILE_M = 8;
  constexpr size_t TILE_N = 8;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(guided)
  for (size_t m = 0; m < M; m += TILE_M) {
    size_t tile_m = std::min(TILE_M, M - m);
    for (size_t n = 0; n < N; n += TILE_N) {
      size_t tile_n = std::min(TILE_N, N - n);
      _hgemm_mk_nk_mn_tile_s8_k16_V2(lhs + m * K, rhs + n * K, bias ? (bias + n) : nullptr,
                                     dst + m * N + n, tile_m, tile_n, M, K, N);
    }
  }
}

void hgemm_mk_kn_mn_V1(const float16_t* __restrict lhs, const float16_t* __restrict rhs,
                       float16_t* __restrict dst, size_t M, size_t K, size_t N,
                       const float16_t* __restrict bias, int threads) {
  /// Micro-kernel interface
  constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
      kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

  const size_t nr = ukernel.get_nr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();

  const size_t rhs_packed_size =
      kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
  const size_t rhs_packed_cols = nr + K * nr;
  const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float16_t));

  auto rhs_packed = new float16_t[rhs_packed_size];

  const size_t lhs_stride = K * sizeof(float16_t);
  const size_t rhs_stride = N * sizeof(float16_t);
  const size_t dst_stride_row = N * sizeof(float16_t);
  const size_t dst_stride_col = sizeof(float16_t);

  // pack once
  kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1, N, K, nr, kr, sr,  // Packing arguments
                                                    rhs_stride,           // RHS stride
                                                    rhs,                  // RHS
                                                    bias,                 // Bias
                                                    nullptr,              // Scale
                                                    rhs_packed,           // RHS packed
                                                    0, nullptr);

  const size_t m_step = ukernel.get_m_step();  // Scheduling along M
  const size_t n_step = ukernel.get_n_step();  // Scheduling along N

#pragma omp parallel for collapse(2) num_threads(threads) schedule(auto) if (threads > 0)
  for (size_t i_m_step = 0; i_m_step < M; i_m_step += m_step) {
    for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step) {
      // Support functions return offset in bytes
      const uint8_t* lhs_ptr =
          (const uint8_t*)lhs + (ukernel.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
      const uint8_t* rhs_ptr =
          (const uint8_t*)rhs_packed + (ukernel.get_rhs_packed_offset(i_n_step, K));
      uint8_t* dst_ptr =
          (uint8_t*)dst + (ukernel.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));

      const size_t actual_m = std::min(M - i_m_step, m_step);
      const size_t actual_n = std::min(N - i_n_step, n_step);

      ukernel.run_matmul(actual_m, actual_n, K,  // Dimensions
                         lhs_ptr,                // LHS
                         lhs_stride,             // LHS stride
                         rhs_ptr,                // RHS packed
                         dst_ptr,                // DST
                         dst_stride_row,         // DST stride (row)
                         dst_stride_col,         // DST stride (col)
                         -65504, 65504           // Min and max for the clamp operation
      );
    }
  }

  delete[] rhs_packed;
}

}  // namespace mllm::arm

#endif