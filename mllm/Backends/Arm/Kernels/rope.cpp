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
#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int s = 0; s < seq_len; ++s) {
    for (int d = 0; d < mid; ++d) {
      float theta = 1.0f / powf(base, 2.0f * (float)d / (float)output_dim);
      theta *= (float)s;
      float sin_value = sinf(theta);
      float cos_value = cosf(theta);
      sin[s * output_dim + d] = sin_value;
      cos[s * output_dim + d] = cos_value;
      sin[s * output_dim + d + mid] = sin_value;
      cos[s * output_dim + d + mid] = cos_value;
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
    const int sin_cos_base = (s + cur_seq_cnt) * dims;

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

void normal_hf_rope_fp16(const float16_t* __restrict X, float16_t* Y, const float* __restrict sin,
                         const float* __restrict cos, int cur_seq_cnt, int seq, int dims,
                         int threads) {
  auto mid_dim = dims / 2;
  for (int s = 0; s < seq; ++s) {
    const float16_t* X_base = X + s * dims;
    float16_t* Y_base = Y + s * dims;

    // Prefill Stage: cur_seq_cnt is always 0
    // Decoding Stage: s is always 0
    const int sin_cos_base = (s + cur_seq_cnt) * dims;

    // Keep in mind that threads may not accelerate this function.
    int d = 0;
    for (; d <= mid_dim - 4; d += 4) {
      // Load input vectors (half-precision)
      const float16_t* X0_ptr = X_base + d;
      const float16_t* X1_ptr = X_base + mid_dim + d;
      float16x4_t x0_half = vld1_f16(X0_ptr);
      float16x4_t x1_half = vld1_f16(X1_ptr);

      // Convert to float32 for computation
      float32x4_t x0 = vcvt_f32_f16(x0_half);
      float32x4_t x1 = vcvt_f32_f16(x1_half);

      // Load sin/cos vectors (float32)
      const float* sin_ptr = sin + sin_cos_base + d;
      const float* cos_ptr = cos + sin_cos_base + d;
      float32x4_t sinv = vld1q_f32(sin_ptr);
      float32x4_t cosv = vld1q_f32(cos_ptr);

      // Compute y0 = x0*cosv - x1*sinv
      float32x4_t y0 = vsubq_f32(vmulq_f32(x0, cosv), vmulq_f32(x1, sinv));

      // Compute y1 = x0*sinv + x1*cosv
      float32x4_t y1 = vaddq_f32(vmulq_f32(x0, sinv), vmulq_f32(x1, cosv));

      // Convert back to half-precision and store
      float16x4_t y0_half = vcvt_f16_f32(y0);
      float16x4_t y1_half = vcvt_f16_f32(y1);
      vst1_f16(Y_base + d, y0_half);
      vst1_f16(Y_base + mid_dim + d, y1_half);
    }

    // Process remaining elements with scalar operations
    for (; d < mid_dim; ++d) {
      // Load half-precision values and convert to float32
      float x0 = static_cast<float>(X_base[d]);
      float x1 = static_cast<float>(X_base[mid_dim + d]);
      float sinv = sin[sin_cos_base + d];
      float cosv = cos[sin_cos_base + d];

      // Compute results in float32
      float y0 = x0 * cosv - x1 * sinv;
      float y1 = x0 * sinv + x1 * cosv;

      // Convert back to half-precision and store
      Y_base[d] = static_cast<float16_t>(y0);
      Y_base[mid_dim + d] = static_cast<float16_t>(y1);
    }
  }
}

void normal_hf_rope_shd(const float* __restrict X, float* Y, const float* __restrict sin,
                        const float* __restrict cos, int cur_seq_cnt, int seq, int dims, int stride,
                        int threads) {
  auto mid_dim = dims / 2;
  for (int s = 0; s < seq; ++s) {
    const float* X_base = X + s * stride;
    float* Y_base = Y + s * stride;

    // Prefill Stage: cur_seq_cnt is always 0
    // Decoding Stage: s is always 0
    const int sin_cos_base = (s + cur_seq_cnt) * dims;

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

void normal_hf_rope_shd_fp16(const float16_t* __restrict X, float16_t* Y,
                             const float* __restrict sin, const float* __restrict cos,
                             int cur_seq_cnt, int seq, int dims, int stride, int threads) {
  auto mid_dim = dims / 2;
  for (int s = 0; s < seq; ++s) {
    const float16_t* X_base = X + s * stride;
    float16_t* Y_base = Y + s * stride;

    // Prefill Stage: cur_seq_cnt is always 0
    // Decoding Stage: s is always 0
    const int sin_cos_base = (s + cur_seq_cnt) * dims;

    // Keep in mind that threads may not accelerate this function.
    int d = 0;
    for (; d <= mid_dim - 4; d += 4) {
      // Load input vectors (half-precision)
      const float16_t* X0_ptr = X_base + d;
      const float16_t* X1_ptr = X_base + mid_dim + d;
      float16x4_t x0_half = vld1_f16(X0_ptr);
      float16x4_t x1_half = vld1_f16(X1_ptr);

      // Convert to float32 for computation
      float32x4_t x0 = vcvt_f32_f16(x0_half);
      float32x4_t x1 = vcvt_f32_f16(x1_half);

      // Load sin/cos vectors (float32)
      const float* sin_ptr = sin + sin_cos_base + d;
      const float* cos_ptr = cos + sin_cos_base + d;
      float32x4_t sinv = vld1q_f32(sin_ptr);
      float32x4_t cosv = vld1q_f32(cos_ptr);

      // Compute y0 = x0*cosv - x1*sinv
      float32x4_t y0 = vsubq_f32(vmulq_f32(x0, cosv), vmulq_f32(x1, sinv));

      // Compute y1 = x0*sinv + x1*cosv
      float32x4_t y1 = vaddq_f32(vmulq_f32(x0, sinv), vmulq_f32(x1, cosv));

      // Convert back to half-precision and store
      float16x4_t y0_half = vcvt_f16_f32(y0);
      float16x4_t y1_half = vcvt_f16_f32(y1);
      vst1_f16(Y_base + d, y0_half);
      vst1_f16(Y_base + mid_dim + d, y1_half);
    }

    // Process remaining elements with scalar operations
    for (; d < mid_dim; ++d) {
      // Load half-precision values and convert to float32
      float x0 = static_cast<float>(X_base[d]);
      float x1 = static_cast<float>(X_base[mid_dim + d]);
      float sinv = sin[sin_cos_base + d];
      float cosv = cos[sin_cos_base + d];

      // Compute results in float32
      float y0 = x0 * cosv - x1 * sinv;
      float y1 = x0 * sinv + x1 * cosv;

      // Convert back to half-precision and store
      Y_base[d] = static_cast<float16_t>(y0);
      Y_base[mid_dim + d] = static_cast<float16_t>(y1);
    }
  }
}

}  // namespace mllm::arm

#endif
