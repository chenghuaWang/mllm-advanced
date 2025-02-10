/**
 * @file rope.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <arm_neon.h>

#include "mllm/Backends/Arm/Kernels/rope.hpp"
#include <cmath>

namespace mllm::arm {

void precompute_normal_hf_sin_cos(int seq_len, int output_dim, float base, float* __restrict sin,
                                  float* __restrict cos, int threads) {
  auto mid = output_dim / 2;
#pragma omp parallel for num_threads(4) schedule(auto) if (threads > 0)
  for (int s = 0; s < seq_len; ++s) {
    for (int d = 0; d < mid; ++d) {
      float theta = 1.0f / powf(base, 2.0f * (float)d / (float)mid);
      theta *= (float)s;
      float sin_value = sinf(theta);
      float cos_value = cosf(theta);
      sin[s * mid + d] = sin_value;
      cos[s * mid + d] = cos_value;
      sin[s * mid + d + mid] = sin_value;
      cos[s * mid + d + mid] = cos_value;
    }
  }
}

void normal_hf_rope(const float* __restrict X, float* Y, const float* __restrict sin,
                    const float* __restrict cos, int cur_seq_cnt, int seq, int dims, int threads) {
  auto mid_dim = dims / 2;
  for (int s = 0; s < seq; ++s) {
    const float* X_base = X + s * dims;
    float* Y_base = Y + s * dims;

    // Prefill Stage: cur_seq_cnt is always 0
    // Decoding Stage: s is always 0
    const int sin_cos_base = (s + cur_seq_cnt) * mid_dim;

    // Keep in mind that threads may not accelerate this function.
    int d = 0;
    for (; d <= mid_dim - 4; d += 4) {
      // Load input vectors
      const float* X0_ptr = X_base + d;
      const float* X1_ptr = X_base + mid_dim + d;
      float32x4_t x0 = vld1q_f32(X0_ptr);
      float32x4_t x1 = vld1q_f32(X1_ptr);

      // Load sin/cos vectors
      const float* sin_ptr = sin + sin_cos_base + d;
      const float* cos_ptr = cos + sin_cos_base + d;
      float32x4_t sinv = vld1q_f32(sin_ptr);
      float32x4_t cosv = vld1q_f32(cos_ptr);

      // Compute y0 = x0*cosv - x1*sinv
      float32x4_t y0 = vsubq_f32(vmulq_f32(x0, cosv), vmulq_f32(x1, sinv));

      // Compute y1 = x0*sinv + x1*cosv
      float32x4_t y1 = vaddq_f32(vmulq_f32(x0, sinv), vmulq_f32(x1, cosv));

      // Store results
      float* Y0_ptr = Y_base + d;
      float* Y1_ptr = Y_base + mid_dim + d;
      vst1q_f32(Y0_ptr, y0);
      vst1q_f32(Y1_ptr, y1);
    }

    // Process remaining elements with scalar operations
    for (; d < mid_dim; ++d) {
      float x0 = X_base[d];
      float x1 = X_base[mid_dim + d];
      float sinv = sin[sin_cos_base + d];
      float cosv = cos[sin_cos_base + d];
      Y_base[d] = x0 * cosv - x1 * sinv;
      Y_base[mid_dim + d] = x0 * sinv + x1 * cosv;
    }
  }
}

}  // namespace mllm::arm

#endif
