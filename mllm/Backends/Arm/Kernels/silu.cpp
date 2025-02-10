/**
 * @file silu.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/silu.hpp"
#include "mllm/Backends/Arm/Kernels/math.hpp"
#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <arm_neon.h>

namespace mllm::arm {

void silu_V1(const float* __restrict X, float* __restrict Y, int len) {
  int i;
  for (i = 0; i <= len - 16; i += 16) {
    float32x4_t x_line_0 = vld1q_f32(X + i);
    float32x4_t ans_line_0 = vmulq_f32(x_line_0, vsigmoid_f32(x_line_0));
    vst1q_f32(Y + i, ans_line_0);

    float32x4_t x_line_1 = vld1q_f32(X + i + 4);
    float32x4_t ans_line_1 = vmulq_f32(x_line_1, vsigmoid_f32(x_line_1));
    vst1q_f32(Y + i + 4, ans_line_1);

    float32x4_t x_line_2 = vld1q_f32(X + i + 8);
    float32x4_t ans_line_2 = vmulq_f32(x_line_2, vsigmoid_f32(x_line_2));
    vst1q_f32(Y + i + 8, ans_line_2);

    float32x4_t x_line_3 = vld1q_f32(X + i + 12);
    float32x4_t ans_line_3 = vmulq_f32(x_line_3, vsigmoid_f32(x_line_3));
    vst1q_f32(Y + i + 12, ans_line_3);
  }
  for (; i <= len - 8; i += 8) {
    float32x4_t x_line_0 = vld1q_f32(X + i);
    float32x4_t ans_line_0 = vmulq_f32(x_line_0, vsigmoid_f32(x_line_0));
    vst1q_f32(Y + i, ans_line_0);

    float32x4_t x_line_1 = vld1q_f32(X + i + 4);
    float32x4_t ans_line_1 = vmulq_f32(x_line_1, vsigmoid_f32(x_line_1));
    vst1q_f32(Y + i + 4, ans_line_1);
  }
  for (; i <= len - 4; i += 4) {
    float32x4_t x_line_0 = vld1q_f32(X + i);
    float32x4_t ans_line_0 = vmulq_f32(x_line_0, vsigmoid_f32(x_line_0));
    vst1q_f32(Y + i, ans_line_0);
  }
  for (; i < len; i++) { Y[i] = X[i] / (1.0f + std::exp(-X[i])); }
}

}  // namespace mllm::arm

#endif
