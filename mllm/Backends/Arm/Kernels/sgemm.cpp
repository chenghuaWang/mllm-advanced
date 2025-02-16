/**
 * @file sgemm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/sgemm.hpp"

#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <algorithm>
#include <arm_neon.h>

// Include micro-kernel variants
#include "kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

namespace mllm::arm {

static inline void _sgemm_mk_nk_mn_tile_s4_k16_V1(const float* __restrict A,
                                                  const float* __restrict B,
                                                  const float* __restrict BIAS, float* __restrict C,
                                                  int ACTUAL_TILE_M, int ACTUAL_TILE_N, int M,
                                                  int K, int N) {
  constexpr int S_TILE = 4;
  constexpr int K_TILE = 16;

  int M_TILE_SIZE = std::min(S_TILE, ACTUAL_TILE_M);
  int N_TILE_SIZE = std::min(S_TILE, ACTUAL_TILE_N);

  float32x4_t c0 = vdupq_n_f32(0);
  float32x4_t c1 = vdupq_n_f32(0);
  float32x4_t c2 = vdupq_n_f32(0);
  float32x4_t c3 = vdupq_n_f32(0);

  if (BIAS) {
    if (N_TILE_SIZE >= 4) {
      c0 = vld1q_f32(BIAS);
      c1 = c0;
      c2 = c0;
      c3 = c0;
    } else {
      float bias_buf[4] = {0};
      for (int n = 0; n < N_TILE_SIZE; ++n) bias_buf[n] = BIAS[n];
      c0 = vld1q_f32(bias_buf);
      c1 = c0;
      c2 = c0;
      c3 = c0;
    }
  }

  for (int k = 0; k < K; k += K_TILE) {
    int k_end = std::min(k + K_TILE, K);
    for (int k_step = k; k_step < k_end; k_step += 4) {
      float32x4_t a[4];
      for (int i = 0; i < M_TILE_SIZE; ++i) { a[i] = vld1q_f32(A + i * K + k_step); }

      float32x4_t b[4];
      for (int j = 0; j < N_TILE_SIZE; ++j) { b[j] = vld1q_f32(B + j * K + k_step); }

      float32x4x2_t tmp0 = vtrnq_f32(b[0], b[1]);
      float32x4x2_t tmp1 = vtrnq_f32(b[2], b[3]);
      float32x4_t bt0 = vcombine_f32(vget_low_f32(tmp0.val[0]), vget_low_f32(tmp1.val[0]));
      float32x4_t bt1 = vcombine_f32(vget_low_f32(tmp0.val[1]), vget_low_f32(tmp1.val[1]));
      float32x4_t bt2 = vcombine_f32(vget_high_f32(tmp0.val[0]), vget_high_f32(tmp1.val[0]));
      float32x4_t bt3 = vcombine_f32(vget_high_f32(tmp0.val[1]), vget_high_f32(tmp1.val[1]));

      for (int i = 0; i < M_TILE_SIZE; ++i) {
        float32x4_t ai = a[i];
        float32x4_t a0 = vdupq_n_f32(vgetq_lane_f32(ai, 0));
        float32x4_t a1 = vdupq_n_f32(vgetq_lane_f32(ai, 1));
        float32x4_t a2 = vdupq_n_f32(vgetq_lane_f32(ai, 2));
        float32x4_t a3 = vdupq_n_f32(vgetq_lane_f32(ai, 3));

        float32x4_t* c = nullptr;
        switch (i) {
          case 0: c = &c0; break;
          case 1: c = &c1; break;
          case 2: c = &c2; break;
          case 3: c = &c3; break;
        }

        *c = vmlaq_f32(*c, a0, bt0);
        *c = vmlaq_f32(*c, a1, bt1);
        *c = vmlaq_f32(*c, a2, bt2);
        *c = vmlaq_f32(*c, a3, bt3);
      }
    }
  }

  for (int i = 0; i < M_TILE_SIZE; ++i) {
    float32x4_t result;
    switch (i) {
      case 0: result = c0; break;
      case 1: result = c1; break;
      case 2: result = c2; break;
      case 3: result = c3; break;
    }

    if (N_TILE_SIZE >= 4) {
      vst1q_f32(C + i * N, result);
    } else {
      switch (N_TILE_SIZE) {
        case 1: vst1q_lane_f32(C + i * N, result, 0); break;
        case 2: vst1_f32(C + i * N, vget_low_f32(result)); break;
        case 3: {
          float32x2_t low = vget_low_f32(result);
          vst1_f32(C + i * N, low);
          vst1q_lane_f32(C + i * N + 2, result, 2);
          break;
        }
      }
    }
  }
}

void sgemm_mk_nk_mn_V1(const float* __restrict lhs, const float* __restrict rhs,
                       float* __restrict dst, int M, int K, int N, const float* __restrict bias,
                       int threads) {
  constexpr int TILE_M = 4;
  constexpr int TILE_N = 4;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(auto) if (threads > 0)
  for (int m = 0; m < M; m += TILE_M) {
    int tile_m = std::min(TILE_M, M - m);
    for (int n = 0; n < N; n += TILE_N) {
      int tile_n = std::min(TILE_N, N - n);
      _sgemm_mk_nk_mn_tile_s4_k16_V1(lhs + m * K, rhs + n * K, bias ? (bias + n) : nullptr,
                                     dst + m * N + n, tile_m, tile_n, M, K, N);
    }
  }
}

void sgemm_mk_kn_mn_V1(const float* __restrict lhs, const float* __restrict rhs,
                       float* __restrict dst, int M, int K, int N, const float* __restrict bias,
                       int threads) {
  constexpr kai_matmul_clamp_f32_f32_f32p_ukernel ukernel{
      kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
      kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla};

  const size_t nr = ukernel.get_nr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();

  const size_t rhs_packed_size =
      kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
  const size_t rhs_packed_cols = nr + K * nr;
  const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float));

  auto rhs_packed = new float[rhs_packed_size];

  const size_t lhs_stride = K * sizeof(float);
  const size_t rhs_stride = N * sizeof(float);
  const size_t dst_stride_row = N * sizeof(float);
  const size_t dst_stride_col = sizeof(float);

  // pack once
  kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1, N, K, nr, kr, sr,  // Packing arguments
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
          (const uint8_t*)lhs + (ukernel.get_lhs_packed_offset(i_m_step, K * sizeof(float)));
      const uint8_t* rhs_ptr =
          (const uint8_t*)rhs_packed + (ukernel.get_rhs_packed_offset(i_n_step, K));
      uint8_t* dst_ptr =
          (uint8_t*)dst + (ukernel.get_dst_offset(i_m_step, i_n_step, N * sizeof(float)));

      const size_t actual_m = std::min(M - i_m_step, m_step);
      const size_t actual_n = std::min(N - i_n_step, n_step);

      ukernel.run_matmul(actual_m, actual_n, K,  // Dimensions
                         lhs_ptr,                // LHS
                         lhs_stride,             // LHS stride
                         rhs_ptr,                // RHS packed
                         dst_ptr,                // DST
                         dst_stride_row,         // DST stride (row)
                         dst_stride_col,         // DST stride (col)
                         -FLT_MAX, FLT_MAX       // Min and max for the clamp operation
      );
    }
  }

  delete[] rhs_packed;
}

}  // namespace mllm::arm

#endif
