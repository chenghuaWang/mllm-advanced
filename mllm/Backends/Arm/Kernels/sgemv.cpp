/**
 * @file sgemv.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
#include <cstring>
#include <arm_neon.h>
#include "mllm/Backends/Arm/Kernels/sgemv.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

void sgemv_1K_NK_V1(const float* __restrict A, const float* __restrict B,
                    const float* __restrict bias, float* __restrict C, int K, int N, int threads) {
  constexpr int K_TILE_SIZE = 16;  // 16 x 4 = 64 bytes, fits to cache line, 4 x 128bits register
  constexpr int N_TILE_SIZE = 4;   // 16 x 128bits register

  MLLM_RT_ASSERT(K % K_TILE_SIZE == 0 && N % N_TILE_SIZE == 0);

  // we assume that C is empty.
  if (bias) {
    std::memcpy(C, bias, N * sizeof(float));
  } else {
    std::memset(C, 0, N * sizeof(float));
  }

  int k_steps = K / K_TILE_SIZE;
  int n_steps = N / N_TILE_SIZE;

  if (threads) {
#pragma omp parallel for num_threads(threads) schedule(auto) collapse(2) reduction(+ : C[ : N])
    for (int k_block_idx = 0; k_block_idx < k_steps; ++k_block_idx) {
      int ks = k_block_idx * K_TILE_SIZE;

      // load 16 fp32 elements from A to A_BLOCK.
      float32x4x4_t A_BLOCK = vld4q_f32(A + ks);  // 4 registers

      for (int n_block_idx = 0; n_block_idx < n_steps; ++n_block_idx) {
        int ns = n_block_idx * N_TILE_SIZE;

        auto B_BLOCK_PTR = B + ns * K + ks;
        if (n_block_idx < n_steps - 1) {
          auto B_BLOCK_NEXT_PTR = B_BLOCK_PTR + 4 * K;
          __builtin_prefetch(B_BLOCK_NEXT_PTR);
          __builtin_prefetch(B_BLOCK_NEXT_PTR + K);
          __builtin_prefetch(B_BLOCK_NEXT_PTR + K * 2);
          __builtin_prefetch(B_BLOCK_NEXT_PTR + K * 3);
        }

        float32x4x4_t B_BLOCK_LINE_0 = vld4q_f32(B_BLOCK_PTR);          // 4 registers
        float32x4x4_t B_BLOCK_LINE_1 = vld4q_f32(B_BLOCK_PTR + K);      // 4 registers
        float32x4x4_t B_BLOCK_LINE_2 = vld4q_f32(B_BLOCK_PTR + K * 2);  // 4 registers
        float32x4x4_t B_BLOCK_LINE_3 = vld4q_f32(B_BLOCK_PTR + K * 3);  // 4 registers

        float32x4x2_t ACC_BLOCK_LINE_0{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers
        float32x4x2_t ACC_BLOCK_LINE_1{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers
        float32x4x2_t ACC_BLOCK_LINE_2{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers
        float32x4x2_t ACC_BLOCK_LINE_3{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers

        // We have 4 registers left right now.

        float32x4_t ACC_BLOCK = vdupq_n_f32(0.f);

        // clang-format off
        // BLOCK LINE 0: 
        // FMA
        ACC_BLOCK_LINE_0.val[0] = vfmaq_f32(ACC_BLOCK_LINE_0.val[0], A_BLOCK.val[0], B_BLOCK_LINE_0.val[0]);
        ACC_BLOCK_LINE_0.val[1] = vfmaq_f32(ACC_BLOCK_LINE_0.val[1], A_BLOCK.val[1], B_BLOCK_LINE_0.val[1]);
        ACC_BLOCK_LINE_0.val[0] = vfmaq_f32(ACC_BLOCK_LINE_0.val[0], A_BLOCK.val[2], B_BLOCK_LINE_0.val[2]);
        ACC_BLOCK_LINE_0.val[1] = vfmaq_f32(ACC_BLOCK_LINE_0.val[1], A_BLOCK.val[3], B_BLOCK_LINE_0.val[3]);
        // Accumulate
        ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_0.val[0], ACC_BLOCK_LINE_0.val[1])), ACC_BLOCK, 0);
  
        // BLOCK LINE 1: 
        // FMA
        ACC_BLOCK_LINE_1.val[0] = vfmaq_f32(ACC_BLOCK_LINE_1.val[0], A_BLOCK.val[0], B_BLOCK_LINE_1.val[0]);
        ACC_BLOCK_LINE_1.val[1] = vfmaq_f32(ACC_BLOCK_LINE_1.val[1], A_BLOCK.val[1], B_BLOCK_LINE_1.val[1]);
        ACC_BLOCK_LINE_1.val[0] = vfmaq_f32(ACC_BLOCK_LINE_1.val[0], A_BLOCK.val[2], B_BLOCK_LINE_1.val[2]);
        ACC_BLOCK_LINE_1.val[1] = vfmaq_f32(ACC_BLOCK_LINE_1.val[1], A_BLOCK.val[3], B_BLOCK_LINE_1.val[3]);
        // Accumulate
        ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_1.val[0], ACC_BLOCK_LINE_1.val[1])), ACC_BLOCK, 1);
  
        // BLOCK LINE 2: 
        // FMA
        ACC_BLOCK_LINE_2.val[0] = vfmaq_f32(ACC_BLOCK_LINE_2.val[0], A_BLOCK.val[0], B_BLOCK_LINE_2.val[0]);
        ACC_BLOCK_LINE_2.val[1] = vfmaq_f32(ACC_BLOCK_LINE_2.val[1], A_BLOCK.val[1], B_BLOCK_LINE_2.val[1]);
        ACC_BLOCK_LINE_2.val[0] = vfmaq_f32(ACC_BLOCK_LINE_2.val[0], A_BLOCK.val[2], B_BLOCK_LINE_2.val[2]);
        ACC_BLOCK_LINE_2.val[1] = vfmaq_f32(ACC_BLOCK_LINE_2.val[1], A_BLOCK.val[3], B_BLOCK_LINE_2.val[3]);
        // Accumulate
        ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_2.val[0], ACC_BLOCK_LINE_2.val[1])), ACC_BLOCK, 2);
  
        // BLOCK LINE 2: 
        // FMA
        ACC_BLOCK_LINE_3.val[0] = vfmaq_f32(ACC_BLOCK_LINE_3.val[0], A_BLOCK.val[0], B_BLOCK_LINE_3.val[0]);
        ACC_BLOCK_LINE_3.val[1] = vfmaq_f32(ACC_BLOCK_LINE_3.val[1], A_BLOCK.val[1], B_BLOCK_LINE_3.val[1]);
        ACC_BLOCK_LINE_3.val[0] = vfmaq_f32(ACC_BLOCK_LINE_3.val[0], A_BLOCK.val[2], B_BLOCK_LINE_3.val[2]);
        ACC_BLOCK_LINE_3.val[1] = vfmaq_f32(ACC_BLOCK_LINE_3.val[1], A_BLOCK.val[3], B_BLOCK_LINE_3.val[3]);
        // Accumulate
        ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_3.val[0], ACC_BLOCK_LINE_3.val[1])), ACC_BLOCK, 3);
        // clang-format on

        float32x4_t ACC_BLOCK_OLD = vld1q_f32(C + ns);
        float32x4_t ACC_BLOCK_NEW = vaddq_f32(ACC_BLOCK, ACC_BLOCK_OLD);
        vst1q_f32(C + ns, ACC_BLOCK_NEW);
      }
    }
    return;
  }

  // gemv iteration.
  for (int k_block_idx = 0; k_block_idx < k_steps; ++k_block_idx) {
    int ks = k_block_idx * K_TILE_SIZE;

    // load 16 fp32 elements from A to A_BLOCK.
    float32x4x4_t A_BLOCK = vld4q_f32(A + ks);  // 4 registers

    for (int n_block_idx = 0; n_block_idx < n_steps; ++n_block_idx) {
      int ns = n_block_idx * N_TILE_SIZE;

      auto B_BLOCK_PTR = B + ns * K + ks;
      if (n_block_idx < n_steps - 1) {
        auto B_BLOCK_NEXT_PTR = B_BLOCK_PTR + 4 * K;
        __builtin_prefetch(B_BLOCK_NEXT_PTR);
        __builtin_prefetch(B_BLOCK_NEXT_PTR + K);
        __builtin_prefetch(B_BLOCK_NEXT_PTR + K * 2);
        __builtin_prefetch(B_BLOCK_NEXT_PTR + K * 3);
      }

      float32x4x4_t B_BLOCK_LINE_0 = vld4q_f32(B_BLOCK_PTR);          // 4 registers
      float32x4x4_t B_BLOCK_LINE_1 = vld4q_f32(B_BLOCK_PTR + K);      // 4 registers
      float32x4x4_t B_BLOCK_LINE_2 = vld4q_f32(B_BLOCK_PTR + K * 2);  // 4 registers
      float32x4x4_t B_BLOCK_LINE_3 = vld4q_f32(B_BLOCK_PTR + K * 3);  // 4 registers

      float32x4x2_t ACC_BLOCK_LINE_0{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers
      float32x4x2_t ACC_BLOCK_LINE_1{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers
      float32x4x2_t ACC_BLOCK_LINE_2{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers
      float32x4x2_t ACC_BLOCK_LINE_3{vdupq_n_f32(0.f), vdupq_n_f32(0.f)};  // 2 registers

      // We have 4 registers left right now.

      float32x4_t ACC_BLOCK = vdupq_n_f32(0.f);

      // clang-format off
      // BLOCK LINE 0: 
      // FMA
      ACC_BLOCK_LINE_0.val[0] = vfmaq_f32(ACC_BLOCK_LINE_0.val[0], A_BLOCK.val[0], B_BLOCK_LINE_0.val[0]);
      ACC_BLOCK_LINE_0.val[1] = vfmaq_f32(ACC_BLOCK_LINE_0.val[1], A_BLOCK.val[1], B_BLOCK_LINE_0.val[1]);
      ACC_BLOCK_LINE_0.val[0] = vfmaq_f32(ACC_BLOCK_LINE_0.val[0], A_BLOCK.val[2], B_BLOCK_LINE_0.val[2]);
      ACC_BLOCK_LINE_0.val[1] = vfmaq_f32(ACC_BLOCK_LINE_0.val[1], A_BLOCK.val[3], B_BLOCK_LINE_0.val[3]);
      // Accumulate
      ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_0.val[0], ACC_BLOCK_LINE_0.val[1])), ACC_BLOCK, 0);

      // BLOCK LINE 1: 
      // FMA
      ACC_BLOCK_LINE_1.val[0] = vfmaq_f32(ACC_BLOCK_LINE_1.val[0], A_BLOCK.val[0], B_BLOCK_LINE_1.val[0]);
      ACC_BLOCK_LINE_1.val[1] = vfmaq_f32(ACC_BLOCK_LINE_1.val[1], A_BLOCK.val[1], B_BLOCK_LINE_1.val[1]);
      ACC_BLOCK_LINE_1.val[0] = vfmaq_f32(ACC_BLOCK_LINE_1.val[0], A_BLOCK.val[2], B_BLOCK_LINE_1.val[2]);
      ACC_BLOCK_LINE_1.val[1] = vfmaq_f32(ACC_BLOCK_LINE_1.val[1], A_BLOCK.val[3], B_BLOCK_LINE_1.val[3]);
      // Accumulate
      ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_1.val[0], ACC_BLOCK_LINE_1.val[1])), ACC_BLOCK, 1);

      // BLOCK LINE 2: 
      // FMA
      ACC_BLOCK_LINE_2.val[0] = vfmaq_f32(ACC_BLOCK_LINE_2.val[0], A_BLOCK.val[0], B_BLOCK_LINE_2.val[0]);
      ACC_BLOCK_LINE_2.val[1] = vfmaq_f32(ACC_BLOCK_LINE_2.val[1], A_BLOCK.val[1], B_BLOCK_LINE_2.val[1]);
      ACC_BLOCK_LINE_2.val[0] = vfmaq_f32(ACC_BLOCK_LINE_2.val[0], A_BLOCK.val[2], B_BLOCK_LINE_2.val[2]);
      ACC_BLOCK_LINE_2.val[1] = vfmaq_f32(ACC_BLOCK_LINE_2.val[1], A_BLOCK.val[3], B_BLOCK_LINE_2.val[3]);
      // Accumulate
      ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_2.val[0], ACC_BLOCK_LINE_2.val[1])), ACC_BLOCK, 2);

      // BLOCK LINE 2: 
      // FMA
      ACC_BLOCK_LINE_3.val[0] = vfmaq_f32(ACC_BLOCK_LINE_3.val[0], A_BLOCK.val[0], B_BLOCK_LINE_3.val[0]);
      ACC_BLOCK_LINE_3.val[1] = vfmaq_f32(ACC_BLOCK_LINE_3.val[1], A_BLOCK.val[1], B_BLOCK_LINE_3.val[1]);
      ACC_BLOCK_LINE_3.val[0] = vfmaq_f32(ACC_BLOCK_LINE_3.val[0], A_BLOCK.val[2], B_BLOCK_LINE_3.val[2]);
      ACC_BLOCK_LINE_3.val[1] = vfmaq_f32(ACC_BLOCK_LINE_3.val[1], A_BLOCK.val[3], B_BLOCK_LINE_3.val[3]);
      // Accumulate
      ACC_BLOCK = vsetq_lane_f32(vaddvq_f32(vaddq_f32(ACC_BLOCK_LINE_3.val[0], ACC_BLOCK_LINE_3.val[1])), ACC_BLOCK, 3);
      // clang-format on

      float32x4_t ACC_BLOCK_OLD = vld1q_f32(C + ns);
      float32x4_t ACC_BLOCK_NEW = vaddq_f32(ACC_BLOCK, ACC_BLOCK_OLD);
      vst1q_f32(C + ns, ACC_BLOCK_NEW);
    }
  }
}

}  // namespace mllm::arm

#endif