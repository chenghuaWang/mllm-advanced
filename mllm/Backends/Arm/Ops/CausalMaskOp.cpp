/**
 * @file CausalMaskOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/CausalMaskOp.hpp"
#include <arm_neon.h>
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmCausalMaskOp::ArmCausalMaskOp(const CausalMaskOpCargo& cargo) : CausalMaskOp(cargo) {}

void ArmCausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ins = inputs[0];
  auto ous = outputs[0];

  auto shape = ins.shape();

  auto B = shape[0];
  auto H = shape[1];
  auto S = shape[2];
  auto D = shape[3];

  switch (ins.dtype()) {
    case kFp32: {
      const float32x4_t mask_val = vdupq_n_f32(-1e10f);

      if (S == 1) {
        for (size_t b = 0; b < B; ++b) {
          for (size_t h = 0; h < H; ++h) {
            auto* i_ptr = ins.offsettedPtr<float>({b, h, 0, 0});
            auto* o_ptr = ous.offsettedPtr<float>({b, h, 0, 0});
            memcpy(o_ptr, i_ptr, D * sizeof(float));
          }
        }
        return;
      }

      for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
          auto* i_ptr = ins.offsettedPtr<float>({b, h, 0, 0});
          auto* o_ptr = ous.offsettedPtr<float>({b, h, 0, 0});

          for (size_t s = 0; s < S; ++s) {
            const size_t row_offset = s * S;
            const size_t copy_count = s + 1;
            const size_t fill_count = S - copy_count;

            if (copy_count > 0) {
              memcpy(o_ptr + row_offset, i_ptr + row_offset, copy_count * sizeof(float));
            }

            float* fill_start = o_ptr + row_offset + copy_count;
            size_t neon_iters = fill_count / 4;
            size_t remainder = fill_count % 4;

            for (size_t i = 0; i < neon_iters; ++i) { vst1q_f32(fill_start + i * 4, mask_val); }
            for (size_t i = 0; i < remainder; ++i) { fill_start[neon_iters * 4 + i] = -1e10f; }
          }
        }
      }
      break;
    }
    case kFp16:
    default: NYI("CausalMaskOp::forward just support fp32 inputs right now");
  }
}

}  // namespace mllm::arm