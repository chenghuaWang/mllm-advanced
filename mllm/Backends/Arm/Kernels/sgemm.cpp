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
}

void sgemm_mk_nk_mn_V1(const float* __restrict lhs, const float* __restrict rhs,
                       float* __restrict dst, int M, int K, int N, const float* __restrict bias,
                       int threads) {
  constexpr int TILE_M = 4;
  constexpr int TILE_N = 4;

  if (threads) {
#pragma omp parallel for collapse(2) num_threads(threads) schedule(auto)
    for (int m = 0; m < M; m += TILE_M) {
      int tile_m = std::min(TILE_M, M - m);
      for (int n = 0; n < N; n += TILE_N) {
        int tile_n = std::min(TILE_N, N - n);
        _sgemm_mk_nk_mn_tile_s4_k16_V1(lhs + m * K, rhs + n * K, bias ? (bias + n) : nullptr,
                                       dst + m * N + n, tile_m, tile_n, M, K, N);
      }
    }
    return;
  }

  for (int m = 0; m < M; m += TILE_M) {
    int tile_m = std::min(TILE_M, M - m);
    for (int n = 0; n < N; n += TILE_N) {
      int tile_n = std::min(TILE_N, N - n);
      _sgemm_mk_nk_mn_tile_s4_k16_V1(lhs + m * K, rhs + n * K, bias ? (bias + n) : nullptr,
                                     dst + m * N + n, tile_m, tile_n, M, K, N);
    }
  }
}

}  // namespace mllm::arm

#endif
