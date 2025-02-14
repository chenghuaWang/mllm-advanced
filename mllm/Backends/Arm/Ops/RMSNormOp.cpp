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
#include <arm_neon.h>
#include <cmath>

namespace mllm::arm {

void ArmRMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto X = inputs[0];
  auto Y = outputs[0];

  auto x_shape = X.shape();
  auto B = x_shape[0];
  auto S = x_shape[1];
  auto H = x_shape[2];
  auto D = x_shape[3];

  switch (X.dtype()) {
    case kFp32: {
      auto w_ptr = weight_.ptr<float>();
#pragma omp parallel for collapse(3) num_threads(cargo_.thread()) if (cargo_.thread() > 0)
      for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
          for (size_t h = 0; h < H; ++h) {
            auto x_ptr = X.offsettedPtr<float>({b, s, h, 0});
            auto y_ptr = Y.offsettedPtr<float>({b, s, h, 0});

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
        }
      }
      break;
    }
    default: NYI("ArmRMSNormOp not support type {} as input", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm
