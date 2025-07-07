/**
 * @file layernorm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cmath>
#include "mllm/Backends/Arm/Kernels/layernorm.hpp"

namespace mllm::arm {
void layernorm_N_fp32(float* __restrict__ Z, const float* __restrict__ X,
                      const float* __restrict__ gamma, const float* __restrict__ beta, size_t N,
                      float eps) {
  if (N == 0) return;

  float sum = 0.0f;
  size_t i = 0;
  float32x4_t vsum = vdupq_n_f32(0.0f);

  for (; i + 3 < N; i += 4) {
    float32x4_t vx = vld1q_f32(X + i);
    vsum = vaddq_f32(vsum, vx);
  }

  sum = vaddvq_f32(vsum);

  for (; i < N; i++) { sum += X[i]; }

  const float mean = sum / N;
  const float32x4_t vmean = vdupq_n_f32(mean);

  float sq_sum = 0.0f;
  i = 0;
  float32x4_t vsq_sum = vdupq_n_f32(0.0f);

  for (; i + 3 < N; i += 4) {
    float32x4_t vx = vld1q_f32(X + i);
    float32x4_t vdiff = vsubq_f32(vx, vmean);
    vsq_sum = vmlaq_f32(vsq_sum, vdiff, vdiff);  // vsq_sum += vdiff * vdiff
  }

  sq_sum = vaddvq_f32(vsq_sum);

  for (; i < N; i++) {
    float diff = X[i] - mean;
    sq_sum += diff * diff;
  }

  const float variance = sq_sum / N;
  const float std_val = 1.0f / sqrtf(variance + eps);
  const float32x4_t vscale = vdupq_n_f32(std_val);

  i = 0;
  if (gamma && beta) {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vnorm = vmulq_f32(vdiff, vscale);

      float32x4_t vgamma = vld1q_f32(gamma + i);
      float32x4_t vbeta = vld1q_f32(beta + i);

      float32x4_t vz = vmlaq_f32(vbeta, vnorm, vgamma);
      vst1q_f32(Z + i, vz);
    }
  } else if (gamma) {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vnorm = vmulq_f32(vdiff, vscale);

      float32x4_t vgamma = vld1q_f32(gamma + i);
      float32x4_t vz = vmulq_f32(vnorm, vgamma);
      vst1q_f32(Z + i, vz);
    }
  } else if (beta) {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vnorm = vmulq_f32(vdiff, vscale);

      float32x4_t vbeta = vld1q_f32(beta + i);
      float32x4_t vz = vaddq_f32(vnorm, vbeta);
      vst1q_f32(Z + i, vz);
    }
  } else {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vz = vmulq_f32(vdiff, vscale);
      vst1q_f32(Z + i, vz);
    }
  }

  for (; i < N; i++) {
    float norm_val = (X[i] - mean) * std_val;
    if (gamma) norm_val *= gamma[i];
    if (beta) norm_val += beta[i];
    Z[i] = norm_val;
  }
}
}  // namespace mllm::arm
