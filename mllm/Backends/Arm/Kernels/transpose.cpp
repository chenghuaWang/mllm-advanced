/**
 * @file transpose.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else
#include <arm_neon.h>
#include "mllm/Backends/Arm/Kernels/transpose.hpp"

namespace mllm::arm {

void transpose_bshd_bhsd(const float* __restrict X, float* __restrict Y, size_t B, size_t S,
                         size_t H, size_t D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int d;
        for (d = 0; d <= D - 4; d += 4) {
          // B, S, H, D
          const float* src_ptr = X + b * S * H * D + s * H * D + h * D + d;

          // B, H, S, D
          float* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          float32x4_t data;
          data = vld1q_f32(src_ptr);
          vst1q_f32(dst_ptr, data);
        }
        for (; d < D; ++d) {
          const float* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          float* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
}

void transpose_bshd_bhsd_fp16(const float16_t* __restrict X, float16_t* __restrict Y, size_t B,
                              size_t S, size_t H, size_t D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int d;
        for (d = 0; d <= D - 8; d += 8) {
          // B, S, H, D
          const float16_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;

          // B, H, S, D
          float16_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          float16x8_t data;
          data = vld1q_f16(src_ptr);
          vst1q_f16(dst_ptr, data);
        }
        for (; d < D; ++d) {
          const float16_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          float16_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
}

}  // namespace mllm::arm
#endif
