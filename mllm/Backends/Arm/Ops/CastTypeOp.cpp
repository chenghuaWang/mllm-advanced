/**
 * @file CastTypeOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <arm_neon.h>
#include "mllm/Backends/Arm/Ops/CastTypeOp.hpp"
#include "mllm/Core/AOps/CastTypeOp.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmCastTypeOp::ArmCastTypeOp(const CastTypeOpCargo& cargo) : CastTypeOp(cargo) {}

void ArmCastTypeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ins = inputs[0];
  auto ous = outputs[0];

  if (ins.isContiguous() && ins.dtype() == kFp32 && ous.dtype() == kFp16) {
    auto ins_ptr = ins.ptr<float>();
    auto ins_len = ins.numel();
    auto ous_ptr = ous.ptr<__fp16>();
    auto ous_len = ous.numel();

    MLLM_RT_ASSERT_EQ(ous_len, ins_len);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    constexpr int NEON_F32_COUNT = 4;
    int neon_loop = ins_len / NEON_F32_COUNT;
    int remain = ins_len % NEON_F32_COUNT;

    for (int i = 0; i < neon_loop; ++i) {
      float32x4_t f32_vec = vld1q_f32(ins_ptr);
      float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
      vst1_f16(ous_ptr, f16_vec);

      ins_ptr += NEON_F32_COUNT;
      ous_ptr += NEON_F32_COUNT;
    }

    for (int i = 0; i < remain; ++i) { *ous_ptr++ = static_cast<__fp16>(*ins_ptr++); }
#else
    for (int i = 0; i < ins_len; ++i) { ous_ptr[i] = static_cast<__fp16>(ins_ptr[i]); }
#endif

    return;
  }

  // This Special case for key_states.to(fp16)
  // [B, H, S, D], not contiguous at S.
  if (!ins.isContiguous() && ins.dtype() == kFp32 && ous.dtype() == kFp16) {
    auto B = ins.shape()[0];
    auto H = ins.shape()[1];
    auto S = ins.shape()[2];
    auto D = ins.shape()[3];
    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t s = 0; s < S; ++s) {
          auto ins_ptr = ins.offsettedPtr<float>({b, h, s, 0});
          auto ous_ptr = ous.offsettedPtr<__fp16>({b, h, s, 0});

          constexpr int NEON_F32_COUNT = 4;
          int neon_loop = D / NEON_F32_COUNT;
          int remain = D % NEON_F32_COUNT;

          for (int i = 0; i < neon_loop; ++i) {
            float32x4_t f32_vec = vld1q_f32(ins_ptr);
            float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
            vst1_f16(ous_ptr, f16_vec);

            ins_ptr += NEON_F32_COUNT;
            ous_ptr += NEON_F32_COUNT;
          }

          for (int i = 0; i < remain; ++i) { *ous_ptr++ = static_cast<__fp16>(*ins_ptr++); }
        }
      }
    }
    return;
  }

  if (ins.isContiguous() && ins.dtype() == kFp16 && ous.dtype() == kFp32) {
    auto ins_ptr = ins.ptr<__fp16>();
    auto ins_len = ins.numel();
    auto ous_ptr = ous.ptr<float>();
    auto ous_len = ous.numel();

    MLLM_RT_ASSERT_EQ(ous_len, ins_len);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    constexpr int NEON_F16_COUNT = 8;
    int neon_loop = ins_len / NEON_F16_COUNT;
    int remain = ins_len % NEON_F16_COUNT;

    for (int i = 0; i < neon_loop; ++i) {
      float16x8_t f16_vec = vld1q_f16(ins_ptr);
      float32x4x2_t f32_vec;
      f32_vec.val[0] = vcvt_f32_f16(vget_low_f16(f16_vec));
      f32_vec.val[1] = vcvt_f32_f16(vget_high_f16(f16_vec));

      vst1q_f32(ous_ptr, f32_vec.val[0]);
      vst1q_f32(ous_ptr + 4, f32_vec.val[1]);

      ins_ptr += NEON_F16_COUNT;
      ous_ptr += NEON_F16_COUNT;
    }

    for (int i = 0; i < remain; ++i) { *ous_ptr++ = static_cast<float>(*ins_ptr++); }
#else
    for (int i = 0; i < ins_len; ++i) { ous_ptr[i] = static_cast<float>(ins_ptr[i]); }
#endif

    return;
  }

  // This Special case for key_states.to(fp32)
  // [B, H, S, D], not contiguous at S.
  if (!ins.isContiguous() && ins.dtype() == kFp16 && ous.dtype() == kFp32) {
    auto B = ins.shape()[0];
    auto H = ins.shape()[1];
    auto S = ins.shape()[2];
    auto D = ins.shape()[3];
    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t s = 0; s < S; ++s) {
          auto ins_ptr = ins.offsettedPtr<__fp16>({b, h, s, 0});
          auto ous_ptr = ous.offsettedPtr<float>({b, h, s, 0});

          constexpr int NEON_F16_COUNT = 8;
          int neon_loop = D / NEON_F16_COUNT;
          int remain = D % NEON_F16_COUNT;

          for (int i = 0; i < neon_loop; ++i) {
            float16x8_t f16_vec = vld1q_f16(ins_ptr);
            float32x4x2_t f32_vec;
            f32_vec.val[0] = vcvt_f32_f16(vget_low_f16(f16_vec));
            f32_vec.val[1] = vcvt_f32_f16(vget_high_f16(f16_vec));

            vst1q_f32(ous_ptr, f32_vec.val[0]);
            vst1q_f32(ous_ptr + 4, f32_vec.val[1]);

            ins_ptr += NEON_F16_COUNT;
            ous_ptr += NEON_F16_COUNT;
          }

          for (int i = 0; i < remain; ++i) { *ous_ptr++ = static_cast<float>(*ins_ptr++); }
        }
      }
    }
    return;
  }
}

}  // namespace mllm::arm