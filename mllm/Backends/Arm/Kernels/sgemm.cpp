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

static inline void _sgemm_mk_kn_mn_tile_s4_k16_V1(const float* __restrict A,
                                                  const float* __restrict B,
                                                  const float* __restrict BIAS, float* __restrict C,
                                                  int ACTUAL_TILE_M, int ACTUAL_TILE_N, int M,
                                                  int K, int N) {}

void sgemm_mk_kn_mn_V1(const float* __restrict lhs, const float* __restrict rhs,
                       float* __restrict dst, int M, int K, int N, const float* __restrict bias,
                       int threads) {
  constexpr int TILE_M = 4;
  constexpr int TILE_N = 4;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(auto) if (threads > 0)
  for (int m = 0; m < M; m += TILE_M) {
    int tile_m = std::min(TILE_M, M - m);
    for (int n = 0; n < N; n += TILE_N) {
      int tile_n = std::min(TILE_N, N - n);
      _sgemm_mk_kn_mn_tile_s4_k16_V1(lhs + m * K, rhs + n * K, bias ? (bias + n) : nullptr,
                                     dst + m * N + n, tile_m, tile_n, M, K, N);
    }
  }
}

}  // namespace mllm::arm

#endif
