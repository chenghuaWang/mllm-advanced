/**
 * @file gelu.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)

 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/math.hpp"
#include "mllm/Backends/Arm/Kernels/gelu.hpp"

namespace mllm::arm {

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N) {
  // 定义常量
  const float32x4_t alpha = vdupq_n_f32(0.044715f);
  const float32x4_t beta = vdupq_n_f32(0.79788456f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t half = vdupq_n_f32(0.5f);

  int i = 0;
  for (; i <= N - 4; i += 4) {
    float32x4_t x = vld1q_f32(X + i);

    float32x4_t x3 = vmulq_f32(x, vmulq_f32(x, x));

    float32x4_t inner = vmlaq_f32(x, alpha, x3);

    float32x4_t scaled = vmulq_f32(beta, inner);

    float32x4_t tanh_val = vtanh_fp32(scaled);

    float32x4_t result = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val));
    vst1q_f32(Z + i, result);
  }

  for (; i < N; i++) {
    float x = X[i];
    float x3 = x * x * x;
    float inner = x + 0.044715f * x3;
    float scaled = 0.79788456f * inner;
    float tanh_val;
    {
      float32x4_t tmp = vtanh_fp32(vdupq_n_f32(scaled));
      tanh_val = vgetq_lane_f32(tmp, 0);
    }
    Z[i] = 0.5f * x * (1.0f + tanh_val);
  }
}

}  // namespace mllm::arm