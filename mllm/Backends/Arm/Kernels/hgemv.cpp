/**
 * @file hgemv.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
#include <cstring>
#include <vector>
#include <arm_neon.h>
#include "mllm/Backends/Arm/Kernels/hgemv.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

void hgemv_1K_NK_V1(const float16_t* __restrict A, const float16_t* __restrict B,
                    const float16_t* __restrict bias, float16_t* __restrict C, int K, int N) {
  MLLM_WARN("The fp16's max value is 65504, which may have precision loss. For fast impl, this "
            "function is not designed for precision sensitive task!!! And this function has not "
            "enable the omp. Multi-threads parallelization is not implemented in this kernel.");

  constexpr int K_TILE_SIZE = 32;  // 32 x 2 = 64 bytes, fits to cache line, 4 x 128bits register
  constexpr int N_TILE_SIZE = 4;   // 16 x 128bits register

  MLLM_RT_ASSERT(K % K_TILE_SIZE == 0 && N % N_TILE_SIZE == 0);

  // we assume that C is empty.
  if (bias) {
    std::memcpy(C, bias, N * sizeof(float16_t));
  } else {
    std::memset(C, 0, N * sizeof(float16_t));
  }

  int k_steps = K / K_TILE_SIZE;
  int n_steps = N / N_TILE_SIZE;

  // gemv iteration.
  for (int k_block_idx = 0; k_block_idx < k_steps; ++k_block_idx) {
    int ks = k_block_idx * K_TILE_SIZE;

    // load 32 fp16 elements from A to A_BLOCK.
    float16x8x4_t A_BLOCK = vld1q_f16_x4(A + ks);  // 4 registers

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

      float16x8x4_t B_BLOCK_LINE_0 = vld1q_f16_x4(B_BLOCK_PTR);          // 4 registers
      float16x8x4_t B_BLOCK_LINE_1 = vld1q_f16_x4(B_BLOCK_PTR + K);      // 4 registers
      float16x8x4_t B_BLOCK_LINE_2 = vld1q_f16_x4(B_BLOCK_PTR + K * 2);  // 4 registers
      float16x8x4_t B_BLOCK_LINE_3 = vld1q_f16_x4(B_BLOCK_PTR + K * 3);  // 4 registers

      float16x8x2_t ACC_BLOCK_LINE_0{vdupq_n_f16(0), vdupq_n_f16(0)};  // 2 registers
      float16x8x2_t ACC_BLOCK_LINE_1{vdupq_n_f16(0), vdupq_n_f16(0)};  // 2 registers
      float16x8x2_t ACC_BLOCK_LINE_2{vdupq_n_f16(0), vdupq_n_f16(0)};  // 2 registers
      float16x8x2_t ACC_BLOCK_LINE_3{vdupq_n_f16(0), vdupq_n_f16(0)};  // 2 registers

      // We have 4 registers left right now.

      float32x4_t ACC_BLOCK = vdupq_n_f32(0.f);

      // clang-format off
      // BLOCK LINE 0: 
      // FMA
      ACC_BLOCK_LINE_0.val[0] = vfmaq_f16(ACC_BLOCK_LINE_0.val[0], A_BLOCK.val[0], B_BLOCK_LINE_0.val[0]);
      ACC_BLOCK_LINE_0.val[1] = vfmaq_f16(ACC_BLOCK_LINE_0.val[1], A_BLOCK.val[1], B_BLOCK_LINE_0.val[1]);
      ACC_BLOCK_LINE_0.val[0] = vfmaq_f16(ACC_BLOCK_LINE_0.val[0], A_BLOCK.val[2], B_BLOCK_LINE_0.val[2]);
      ACC_BLOCK_LINE_0.val[1] = vfmaq_f16(ACC_BLOCK_LINE_0.val[1], A_BLOCK.val[3], B_BLOCK_LINE_0.val[3]);
      // Accumulate
      float16x8_t ACC_FP16_LINE_0 = vaddq_f16(ACC_BLOCK_LINE_0.val[0], ACC_BLOCK_LINE_0.val[1]);
      float32x4_t ACC_FP32_LINE_0_0 = vcvt_f32_f16(vget_low_f16(ACC_FP16_LINE_0));
      float32x4_t ACC_FP32_LINE_0_1 = vcvt_f32_f16(vget_high_f16(ACC_FP16_LINE_0));
      float32_t ACC_LINE_0 = vaddvq_f32(ACC_FP32_LINE_0_0) + vaddvq_f32(ACC_FP32_LINE_0_1);
      ACC_BLOCK = vsetq_lane_f32(ACC_LINE_0, ACC_BLOCK, 0);

      // BLOCK LINE 1: 
      // FMA
      ACC_BLOCK_LINE_1.val[0] = vfmaq_f16(ACC_BLOCK_LINE_1.val[0], A_BLOCK.val[0], B_BLOCK_LINE_1.val[0]);
      ACC_BLOCK_LINE_1.val[1] = vfmaq_f16(ACC_BLOCK_LINE_1.val[1], A_BLOCK.val[1], B_BLOCK_LINE_1.val[1]);
      ACC_BLOCK_LINE_1.val[0] = vfmaq_f16(ACC_BLOCK_LINE_1.val[0], A_BLOCK.val[2], B_BLOCK_LINE_1.val[2]);
      ACC_BLOCK_LINE_1.val[1] = vfmaq_f16(ACC_BLOCK_LINE_1.val[1], A_BLOCK.val[3], B_BLOCK_LINE_1.val[3]);
      // Accumulate
      float16x8_t ACC_FP16_LINE_1 = vaddq_f16(ACC_BLOCK_LINE_1.val[0], ACC_BLOCK_LINE_1.val[1]);
      float32x4_t ACC_FP32_LINE_1_0 = vcvt_f32_f16(vget_low_f16(ACC_FP16_LINE_1));
      float32x4_t ACC_FP32_LINE_1_1 = vcvt_f32_f16(vget_high_f16(ACC_FP16_LINE_1));
      float32_t ACC_LINE_1 = vaddvq_f32(ACC_FP32_LINE_1_0) + vaddvq_f32(ACC_FP32_LINE_1_1);
      ACC_BLOCK = vsetq_lane_f32(ACC_LINE_1, ACC_BLOCK, 1);

      // BLOCK LINE 2: 
      // FMA
      ACC_BLOCK_LINE_2.val[0] = vfmaq_f16(ACC_BLOCK_LINE_2.val[0], A_BLOCK.val[0], B_BLOCK_LINE_2.val[0]);
      ACC_BLOCK_LINE_2.val[1] = vfmaq_f16(ACC_BLOCK_LINE_2.val[1], A_BLOCK.val[1], B_BLOCK_LINE_2.val[1]);
      ACC_BLOCK_LINE_2.val[0] = vfmaq_f16(ACC_BLOCK_LINE_2.val[0], A_BLOCK.val[2], B_BLOCK_LINE_2.val[2]);
      ACC_BLOCK_LINE_2.val[1] = vfmaq_f16(ACC_BLOCK_LINE_2.val[1], A_BLOCK.val[3], B_BLOCK_LINE_2.val[3]);
      // Accumulate
      float16x8_t ACC_FP16_LINE_2 = vaddq_f16(ACC_BLOCK_LINE_2.val[0], ACC_BLOCK_LINE_2.val[1]);
      float32x4_t ACC_FP32_LINE_2_0 = vcvt_f32_f16(vget_low_f16(ACC_FP16_LINE_2));
      float32x4_t ACC_FP32_LINE_2_1 = vcvt_f32_f16(vget_high_f16(ACC_FP16_LINE_2));
      float32_t ACC_LINE_2 = vaddvq_f32(ACC_FP32_LINE_2_0) + vaddvq_f32(ACC_FP32_LINE_2_1);
      ACC_BLOCK = vsetq_lane_f32(ACC_LINE_2, ACC_BLOCK, 2);

      // BLOCK LINE 3: 
      // FMA
      ACC_BLOCK_LINE_3.val[0] = vfmaq_f16(ACC_BLOCK_LINE_3.val[0], A_BLOCK.val[0], B_BLOCK_LINE_3.val[0]);
      ACC_BLOCK_LINE_3.val[1] = vfmaq_f16(ACC_BLOCK_LINE_3.val[1], A_BLOCK.val[1], B_BLOCK_LINE_3.val[1]);
      ACC_BLOCK_LINE_3.val[0] = vfmaq_f16(ACC_BLOCK_LINE_3.val[0], A_BLOCK.val[2], B_BLOCK_LINE_3.val[2]);
      ACC_BLOCK_LINE_3.val[1] = vfmaq_f16(ACC_BLOCK_LINE_3.val[1], A_BLOCK.val[3], B_BLOCK_LINE_3.val[3]);
      // Accumulate
      float16x8_t ACC_FP16_LINE_3 = vaddq_f16(ACC_BLOCK_LINE_3.val[0], ACC_BLOCK_LINE_3.val[1]);
      float32x4_t ACC_FP32_LINE_3_0 = vcvt_f32_f16(vget_low_f16(ACC_FP16_LINE_3));
      float32x4_t ACC_FP32_LINE_3_1 = vcvt_f32_f16(vget_high_f16(ACC_FP16_LINE_3));
      float32_t ACC_LINE_3 = vaddvq_f32(ACC_FP32_LINE_3_0) + vaddvq_f32(ACC_FP32_LINE_3_1);
      ACC_BLOCK = vsetq_lane_f32(ACC_LINE_3, ACC_BLOCK, 3);
      // clang-format on

      float16x4_t ACC_BLOCK_OLD = vld1_f16(C + ns);
      float16x4_t ACC_BLOCK_NEW = vadd_f16(vcvt_f16_f32(ACC_BLOCK), ACC_BLOCK_OLD);
      vst1_f16(C + ns, ACC_BLOCK_NEW);
    }
  }
}

void hgemv_1K_NK_V2_HP(const float16_t* __restrict A, const float16_t* __restrict B,
                       const float16_t* __restrict bias, float16_t* __restrict C, int K, int N) {
  constexpr int K_TILE_SIZE = 32;
  constexpr int N_TILE_SIZE = 4;

  MLLM_RT_ASSERT(K % K_TILE_SIZE == 0 && N % N_TILE_SIZE == 0);

  std::vector<float> C_fp32(N, 0.f);
  if (bias) {
    for (int i = 0; i < N; ++i) { C_fp32[i] = static_cast<float>(bias[i]); }
  }

  int k_steps = K / K_TILE_SIZE;
  int n_steps = N / N_TILE_SIZE;

  for (int k_block_idx = 0; k_block_idx < k_steps; ++k_block_idx) {
    const int ks = k_block_idx * K_TILE_SIZE;
    const float16_t* A_ptr = A + ks;

    float16x8x4_t A_BLOCK = vld1q_f16_x4(A_ptr);

    for (int n_block_idx = 0; n_block_idx < n_steps; ++n_block_idx) {
      const int ns = n_block_idx * N_TILE_SIZE;
      const float16_t* B_ptr = B + ns * K + ks;

      if (n_block_idx < n_steps - 1) {
        const float16_t* next_B_ptr = B_ptr + N_TILE_SIZE * K;
        __builtin_prefetch(next_B_ptr);
        __builtin_prefetch(next_B_ptr + K);
        __builtin_prefetch(next_B_ptr + 2 * K);
        __builtin_prefetch(next_B_ptr + 3 * K);
      }

      float16x8x4_t B_row0 = vld1q_f16_x4(B_ptr);
      float16x8x4_t B_row1 = vld1q_f16_x4(B_ptr + K);
      float16x8x4_t B_row2 = vld1q_f16_x4(B_ptr + 2 * K);
      float16x8x4_t B_row3 = vld1q_f16_x4(B_ptr + 3 * K);

      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(A_BLOCK.val[i]));
        float32x4_t a1 = vcvt_f32_f16(vget_high_f16(A_BLOCK.val[i]));

        float32x4_t b0_row0 = vcvt_f32_f16(vget_low_f16(B_row0.val[i]));
        float32x4_t b1_row0 = vcvt_f32_f16(vget_high_f16(B_row0.val[i]));
        acc0 = vfmaq_f32(acc0, a0, b0_row0);
        acc0 = vfmaq_f32(acc0, a1, b1_row0);

        float32x4_t b0_row1 = vcvt_f32_f16(vget_low_f16(B_row1.val[i]));
        float32x4_t b1_row1 = vcvt_f32_f16(vget_high_f16(B_row1.val[i]));
        acc1 = vfmaq_f32(acc1, a0, b0_row1);
        acc1 = vfmaq_f32(acc1, a1, b1_row1);

        float32x4_t b0_row2 = vcvt_f32_f16(vget_low_f16(B_row2.val[i]));
        float32x4_t b1_row2 = vcvt_f32_f16(vget_high_f16(B_row2.val[i]));
        acc2 = vfmaq_f32(acc2, a0, b0_row2);
        acc2 = vfmaq_f32(acc2, a1, b1_row2);

        float32x4_t b0_row3 = vcvt_f32_f16(vget_low_f16(B_row3.val[i]));
        float32x4_t b1_row3 = vcvt_f32_f16(vget_high_f16(B_row3.val[i]));
        acc3 = vfmaq_f32(acc3, a0, b0_row3);
        acc3 = vfmaq_f32(acc3, a1, b1_row3);
      }

      float32x4_t sum0 = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
      float32x4_t old_val = vld1q_f32(C_fp32.data() + ns);
      vst1q_f32(C_fp32.data() + ns, vaddq_f32(old_val, sum0));
    }
  }

  for (int i = 0; i < N; i += 4) {
    float32x4_t res = vld1q_f32(C_fp32.data() + i);
    vst1_f16(C + i, vcvt_f16_f32(res));
  }
}

}  // namespace mllm::arm

#endif