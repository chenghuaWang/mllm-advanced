/**
 * @file RMSNorm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/RMSNormOp.hpp"
#include "mllm/Backends/Arm/Kernels/math.hpp"
#include "mllm/Utils/Common.hpp"
#include <arm_neon.h>
#include <cmath>

namespace mllm::arm {

ArmRMSNormOp::ArmRMSNormOp(const RMSNormOpCargo& cargo) : RMSNormOp(cargo) {}

void ArmRMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // should support [B, S, H * D] and [B, S, H, D]
  auto X = inputs[0];
  auto Y = outputs[0];

  MLLM_RT_ASSERT_EQ(X.shape().size(), Y.shape().size());

  auto x_shape = X.shape();
  auto D = x_shape[x_shape.size() - 1];
  size_t other_dim_size = 1;
  for (size_t i = 0; i < x_shape.size() - 1; ++i) { other_dim_size *= x_shape[i]; }

  switch (X.dtype()) {
    case kFp32: {
      auto w_ptr = weight_.ptr<float>();
#pragma omp parallel for num_threads(cargo_.thread()) if (cargo_.thread() > 0)
      for (size_t other_dim = 0; other_dim < other_dim_size; ++other_dim) {
        auto x_ptr = X.ptr<float>() + other_dim * D;
        auto y_ptr = Y.ptr<float>() + other_dim * D;

        // pass 1
        const float rms = 1.f / std::sqrtf(vsquare_mean_fp32(x_ptr, D) + cargo_.epsilon);

        // pass 2
        if (cargo_.add_unit_offset) {
          float32x4_t ones = vdupq_n_f32(1.f);
          int d;
          for (d = 0; d <= D - 4; ++d) {
            float32x4_t tmp_x = vld1q_f32(x_ptr + d);
            float32x4_t multiplier = vld1q_f32(w_ptr + d);
            multiplier = vaddq_f32(multiplier, ones);
            multiplier = vmulq_n_f32(multiplier, rms);
            float32x4_t tmp_Y = vmulq_f32(tmp_x, multiplier);
            vst1q_f32(y_ptr + d, tmp_Y);
          }
          for (; d < D; ++d) {
            float tmp_X = x_ptr[d];
            float multiplier = w_ptr[d] + 1.f;
            y_ptr[d] = tmp_X * rms * multiplier;
          }
        } else {
          int d;
          for (d = 0; d <= D - 4; ++d) {
            float32x4_t tmp_x = vld1q_f32(x_ptr + d);
            float32x4_t multiplier = vld1q_f32(w_ptr + d);
            multiplier = vmulq_n_f32(multiplier, rms);
            float32x4_t tmp_Y = vmulq_f32(tmp_x, multiplier);
            vst1q_f32(y_ptr + d, tmp_Y);
          }
          for (; d < D; ++d) {
            float tmp_X = x_ptr[d];
            float multiplier = w_ptr[d];
            y_ptr[d] = tmp_X * rms * multiplier;
          }
        }
      }
      break;
    }
    case kFp16: {
      auto w_ptr = weight_.ptr<float16_t>();
#pragma omp parallel for num_threads(cargo_.thread()) if (cargo_.thread() > 0)
      for (size_t other_dim = 0; other_dim < other_dim_size; ++other_dim) {
        auto x_ptr = X.ptr<float16_t>() + other_dim * D;
        auto y_ptr = Y.ptr<float16_t>() + other_dim * D;

        // pass 1: compute RMS scaling factor
        float mean_square = vsquare_mean_fp16(x_ptr, D);
        const float rms_float = 1.f / std::sqrtf(mean_square + cargo_.epsilon);
        float16_t rms_fp16 = static_cast<float16_t>(rms_float);
        float16x8_t rms_vec = vdupq_n_f16(rms_fp16);

        // pass 2: apply scaling with weight
        if (cargo_.add_unit_offset) {
          float16x8_t ones = vdupq_n_f16(1.0f);
          int d = 0;
          for (; d <= D - 8; d += 8) {
            float16x8_t tmp_x = vld1q_f16(x_ptr + d);
            float16x8_t multiplier = vld1q_f16(w_ptr + d);
            multiplier = vaddq_f16(multiplier, ones);
            multiplier = vmulq_f16(multiplier, rms_vec);
            float16x8_t tmp_y = vmulq_f16(tmp_x, multiplier);
            vst1q_f16(y_ptr + d, tmp_y);
          }
          for (; d < D; ++d) { y_ptr[d] = x_ptr[d] * rms_fp16 * (w_ptr[d] + 1.0f); }
        } else {
          int d = 0;
          for (; d <= D - 8; d += 8) {
            float16x8_t tmp_x = vld1q_f16(x_ptr + d);
            float16x8_t multiplier = vld1q_f16(w_ptr + d);
            multiplier = vmulq_f16(multiplier, rms_vec);
            float16x8_t tmp_y = vmulq_f16(tmp_x, multiplier);
            vst1q_f16(y_ptr + d, tmp_y);
          }
          for (; d < D; ++d) { y_ptr[d] = x_ptr[d] * rms_fp16 * w_ptr[d]; }
        }
      }
      break;
    }
    default: NYI("ArmRMSNormOp not support type {} as input", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm
