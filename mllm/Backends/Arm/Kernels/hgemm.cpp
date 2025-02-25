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

static inline void _hgemm_mk_nk_mn_tile_s4_k16_V1(const __fp16* __restrict A,
                                                  const __fp16* __restrict B,
                                                  const __fp16* __restrict BIAS,
                                                  __fp16* __restrict C, int ACTUAL_TILE_M,
                                                  int ACTUAL_TILE_N, int M, int K, int N) {
  constexpr int S_TILE = 4;
  constexpr int K_TILE = 16;

  int M_TILE_SIZE = std::min(S_TILE, ACTUAL_TILE_M);
  int N_TILE_SIZE = std::min(S_TILE, ACTUAL_TILE_N);

  const __fp16 FP16_MIN = -65504.0f;
  const __fp16 FP16_MAX = 65504.0f;
  const float16x4_t clamp_min = vdup_n_f16(FP16_MIN);
  const float16x4_t clamp_max = vdup_n_f16(FP16_MAX);

  float16x4_t c0 = vdup_n_f16(0);
  float16x4_t c1 = vdup_n_f16(0);
  float16x4_t c2 = vdup_n_f16(0);
  float16x4_t c3 = vdup_n_f16(0);

  if (BIAS) {
    if (N_TILE_SIZE >= 4) {
      c0 = vld1_f16(BIAS);
      c1 = c0;
      c2 = c0;
      c3 = c0;
    } else {
      __fp16 bias_buf[4] = {0};
      for (int n = 0; n < N_TILE_SIZE; ++n) bias_buf[n] = BIAS[n];
      c0 = vld1_f16(bias_buf);
      c1 = c0;
      c2 = c0;
      c3 = c0;
    }
  }

  for (int k = 0; k < K; k += K_TILE) {
    int k_end = std::min(k + K_TILE, K);
    for (int k_step = k; k_step < k_end; k_step += 4) {
      float16x4_t a[4];
      for (int i = 0; i < M_TILE_SIZE; ++i) { a[i] = vld1_f16(A + i * K + k_step); }

      float16x4_t b[4];
      for (int j = 0; j < N_TILE_SIZE; ++j) { b[j] = vld1_f16(B + j * K + k_step); }

      // Transpose B matrix
      float16x4x2_t tmp0 = vtrn_f16(b[0], b[1]);
      float16x4x2_t tmp1 = vtrn_f16(b[2], b[3]);

      // Construct transposed vectors
      __fp16 bt0_data[4] = {vget_lane_f16(tmp0.val[0], 0), vget_lane_f16(tmp0.val[0], 1),
                            vget_lane_f16(tmp1.val[0], 0), vget_lane_f16(tmp1.val[0], 1)};
      __fp16 bt1_data[4] = {vget_lane_f16(tmp0.val[1], 0), vget_lane_f16(tmp0.val[1], 1),
                            vget_lane_f16(tmp1.val[1], 0), vget_lane_f16(tmp1.val[1], 1)};
      __fp16 bt2_data[4] = {vget_lane_f16(tmp0.val[0], 2), vget_lane_f16(tmp0.val[0], 3),
                            vget_lane_f16(tmp1.val[0], 2), vget_lane_f16(tmp1.val[0], 3)};
      __fp16 bt3_data[4] = {vget_lane_f16(tmp0.val[1], 2), vget_lane_f16(tmp0.val[1], 3),
                            vget_lane_f16(tmp1.val[1], 2), vget_lane_f16(tmp1.val[1], 3)};

      float16x4_t bt0 = vld1_f16(bt0_data);
      float16x4_t bt1 = vld1_f16(bt1_data);
      float16x4_t bt2 = vld1_f16(bt2_data);
      float16x4_t bt3 = vld1_f16(bt3_data);

      for (int i = 0; i < M_TILE_SIZE; ++i) {
        float16x4_t ai = a[i];
        float16x4_t a0 = vdup_n_f16(vget_lane_f16(ai, 0));
        float16x4_t a1 = vdup_n_f16(vget_lane_f16(ai, 1));
        float16x4_t a2 = vdup_n_f16(vget_lane_f16(ai, 2));
        float16x4_t a3 = vdup_n_f16(vget_lane_f16(ai, 3));

        float16x4_t* c = nullptr;
        switch (i) {
          case 0: c = &c0; break;
          case 1: c = &c1; break;
          case 2: c = &c2; break;
          case 3: c = &c3; break;
        }

        *c = vfma_f16(*c, a0, bt0);
        *c = vfma_f16(*c, a1, bt1);
        *c = vfma_f16(*c, a2, bt2);
        *c = vfma_f16(*c, a3, bt3);
      }
    }
  }

  for (int i = 0; i < M_TILE_SIZE; ++i) {
    float16x4_t result;
    switch (i) {
      case 0: result = c0; break;
      case 1: result = c1; break;
      case 2: result = c2; break;
      case 3: result = c3; break;
    }

    result = vmax_f16(result, clamp_min);  // lower bound
    result = vmin_f16(result, clamp_max);  // upper bound

    if (N_TILE_SIZE >= 4) {
      vst1_f16(C + i * N, result);
    } else {
      switch (N_TILE_SIZE) {
        case 1: vst1_lane_f16(C + i * N, result, 0); break;
        case 2: {
          vst1_lane_f16(C + i * N, result, 0);
          vst1_lane_f16(C + i * N + 1, result, 1);
          break;
        }
        case 3: {
          vst1_lane_f16(C + i * N, result, 0);
          vst1_lane_f16(C + i * N + 1, result, 1);
          vst1_lane_f16(C + i * N + 2, result, 2);
          break;
        }
      }
    }
  }
}

void hgemm_mk_nk_mn_V1(const float16_t* __restrict lhs, const float16_t* __restrict rhs,
                       float16_t* __restrict dst, size_t M, size_t K, size_t N,
                       const float16_t* __restrict bias, int threads) {
  constexpr size_t TILE_M = 4;
  constexpr size_t TILE_N = 4;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(guided)
  for (size_t m = 0; m < M; m += TILE_M) {
    size_t tile_m = std::min(TILE_M, M - m);
    for (size_t n = 0; n < N; n += TILE_N) {
      size_t tile_n = std::min(TILE_N, N - n);
      _hgemm_mk_nk_mn_tile_s4_k16_V1(lhs + m * K, rhs + n * K, bias ? (bias + n) : nullptr,
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