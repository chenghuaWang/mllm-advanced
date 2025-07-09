/**
 * @file MultimodalRoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#include "mllm/Backends/Arm/Ops/MultimodalRoPEOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"
#include <arm_neon.h>

namespace mllm::arm {

Tensor Qwen2VLMultimodalRoPEOpImpl::makeInvFreq(int output_dim, float rope_theta) {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFp32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) {
    inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim);
  }
  return inv_freq;
}

std::pair<Tensor, Tensor> Qwen2VLMultimodalRoPEOpImpl::makePositionEmbedding(
    Tensor& position_ids, Tensor& inv_freq, int seq_len, int output_dim,
    std::vector<int32_t>& mrope_section) {
  // Position ids shape is [3, 1, seq]
  MLLM_RT_ASSERT_EQ(position_ids.shape().size(), 3);
  MLLM_RT_ASSERT_EQ(position_ids.shape()[1], 1);  // Batch size is always 1.

  // [3, seq, dim]
  Tensor tmp_sin = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();
  Tensor tmp_cos = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();

  for (int b = 0; b < 3; ++b) {
    for (int d = 0; d < inv_freq.shape()[0]; ++d) {
      for (int s = 0; s < position_ids.shape()[2]; ++s) {
        auto value = inv_freq.ptr<float>()[d] * (*position_ids.offsettedPtr<float>({b, 0, s}));
        *tmp_cos.offsettedPtr<float>({b, s, d}) = cosf(value);
        *tmp_cos.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = cosf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d}) = sinf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = sinf(value);
      }
    }
  }

  Tensor sin = Tensor::nil();
  Tensor cos = Tensor::nil();

  // mrope is always [16, 24, 24]
  if (!mrope_section.empty()) {
    int num_rows = tmp_sin.shape()[1];
    int num_cols = tmp_sin.shape()[2];

    sin = Tensor::empty({num_rows, num_cols}, kFp32, kCPU).alloc();
    cos = Tensor::empty({num_rows, num_cols}, kFp32, kCPU).alloc();

    std::vector<int> start_cols;
    int current_start = 0;
    start_cols.push_back(current_start);
    for (int s : mrope_section) {
      current_start += s;
      start_cols.push_back(current_start);
    }

    for (int j = 0; j < mrope_section.size(); ++j) {
      int layer = j % 3;
      int s_j = mrope_section[j];
      int start_col_in = start_cols[j];
      int start_col_out = start_cols[j];
      for (int row = 0; row < num_rows; ++row) {
        // Process cos
        auto in_cos_row_ptr = tmp_cos.offsettedPtr<float>({layer, row, 0});
        auto out_cos_row_ptr = cos.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) {
          out_cos_row_ptr[start_col_out + c] = in_cos_row_ptr[start_col_in + c];
        }

        // Process sin
        auto in_sin_row_ptr = tmp_sin.offsettedPtr<float>({layer, row, 0});
        auto out_sin_row_ptr = sin.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) {
          out_sin_row_ptr[start_col_out + c] = in_sin_row_ptr[start_col_in + c];
        }
      }
    }

  } else {
    sin = tmp_sin;
    cos = tmp_cos;
  }

  return {sin, cos};
}

void Qwen2VLMultimodalRoPEOpImpl::forward(const std::vector<Tensor>& inputs,
                                          std::vector<Tensor>& outputs, Tensor& sin, Tensor& cos) {
  auto activation = inputs[0];
  auto out = outputs[0];

  // Activation must in BSHD layout
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  auto B = activation.shape()[0];
  auto S = activation.shape()[1];
  auto H = activation.shape()[2];
  auto D = activation.shape()[3];

  int32_t partial_dimension = D;
  int32_t half = D / 2;

  switch (activation.dtype()) {
    case kFp32: {
      for (int n = 0; n < B; ++n) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            float* act_ptr = activation.offsettedPtr<float>({n, s, h, 0});
            float* out_ptr = out.offsettedPtr<float>({n, s, h, 0});
            const float* sin_ptr = sin.offsettedPtr<float>({s, 0});
            const float* cos_ptr = cos.offsettedPtr<float>({s, 0});

            // Vectorized processing (4 elements per iteration)
            int d = 0;
            constexpr int step = 4;
            for (; d <= half - step; d += step) {
              // Load activation blocks
              float32x4_t act_front = vld1q_f32(act_ptr + d);
              float32x4_t act_back = vld1q_f32(act_ptr + d + half);

              // Load sin/cos values
              float32x4_t sin_vec = vld1q_f32(sin_ptr + d);
              float32x4_t cos_vec = vld1q_f32(cos_ptr + d);

              // Compute rotated values
              float32x4_t out_front =
                  vsubq_f32(vmulq_f32(act_front, cos_vec), vmulq_f32(act_back, sin_vec));

              float32x4_t out_back =
                  vaddq_f32(vmulq_f32(act_front, sin_vec), vmulq_f32(act_back, cos_vec));

              // Store results
              vst1q_f32(out_ptr + d, out_front);
              vst1q_f32(out_ptr + d + half, out_back);
            }

            // Process remaining elements
            for (; d < half; ++d) {
              float in_val = act_ptr[d];
              float in_val2 = act_ptr[d + half];
              float sin_val = sin_ptr[d];
              float cos_val = cos_ptr[d];

              out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
              out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
            }
          }
        }
      }
      break;
    }
    case kFp16: {
      for (int n = 0; n < B; ++n) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            float16_t* act_ptr = activation.offsettedPtr<float16_t>({n, s, h, 0});
            float16_t* out_ptr = out.offsettedPtr<float16_t>({n, s, h, 0});
            const float16_t* sin_ptr = sin.offsettedPtr<float16_t>({s, 0});
            const float16_t* cos_ptr = cos.offsettedPtr<float16_t>({s, 0});

            // Vectorized processing (8 elements per iteration)
            int d = 0;
            constexpr int step = 8;
            for (; d <= half - step; d += step) {
              // Load activation blocks
              float16x8_t act_front = vld1q_f16(act_ptr + d);
              float16x8_t act_back = vld1q_f16(act_ptr + d + half);

              // Load sin/cos values
              float16x8_t sin_vec = vld1q_f16(sin_ptr + d);
              float16x8_t cos_vec = vld1q_f16(cos_ptr + d);

              // Compute rotated values
              float16x8_t out_front =
                  vsubq_f16(vmulq_f16(act_front, cos_vec), vmulq_f16(act_back, sin_vec));

              float16x8_t out_back =
                  vaddq_f16(vmulq_f16(act_front, sin_vec), vmulq_f16(act_back, cos_vec));

              // Store results
              vst1q_f16(out_ptr + d, out_front);
              vst1q_f16(out_ptr + d + half, out_back);
            }

            // Process remaining elements
            for (; d < half; ++d) {
              float in_val = static_cast<float>(act_ptr[d]);
              float in_val2 = static_cast<float>(act_ptr[d + half]);
              float sin_val = static_cast<float>(sin_ptr[d]);
              float cos_val = static_cast<float>(cos_ptr[d]);

              out_ptr[d] = static_cast<float16_t>(in_val * cos_val - in_val2 * sin_val);
              out_ptr[d + half] = static_cast<float16_t>(in_val * sin_val + in_val2 * cos_val);
            }
          }
        }
      }
      break;
    }
    default: {
      NYI("Qwen2VLMultimodalRoPEOpImpl::forward not support this dtype")
      break;
    }
  }
}

ArmMultimodalRoPEOp::ArmMultimodalRoPEOp(const MultimodalRoPEOpCargo& cargo)
    : MultimodalRoPEOp(cargo) {}

void ArmMultimodalRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& ctx = MllmEngineCtx::instance();

  auto& activation = inputs[0];

  // Expect 2 inputs:
  // Pos 1: activations
  // Pos 2: position_ids
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);

  // Input must be [B, S, H, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  auto position_ids = inputs[1];
  auto out = outputs[0];

  switch (cargo_.type) {
    case MultimodalRoPEOpCargoType::kQwen2VL: {
      auto impl = Qwen2VLMultimodalRoPEOpImpl();

      Tensor sin = Tensor::nil();
      Tensor cos = Tensor::nil();

      // Gen/load sin and cos.
      if (ctx.mem()->hasGlobalTensor("__qwen2vl_model_rope_sin")
          && ctx.mem()->hasGlobalTensor("__qwen2vl_model_rope_cos")) {
        sin = ctx.mem()->getGlobalTensor("__qwen2vl_model_rope_sin");
        cos = ctx.mem()->getGlobalTensor("__qwen2vl_model_rope_cos");
      } else {
        auto inv_freq = impl.makeInvFreq(activation.shape()[3], cargo_.rope_theta);
        auto [_sin, _cos] =
            impl.makePositionEmbedding(position_ids, inv_freq, cargo_.max_position_embeddings,
                                       activation.shape()[3], cargo_.mrope_section);
        sin = _sin;
        cos = _cos;

        sin = sin.setMemType(kGlobal).setName("__qwen2vl_model_rope_sin");
        cos = cos.setMemType(kGlobal).setName("__qwen2vl_model_rope_cos");

        ctx.mem()->regGlobalTensor(sin);
        ctx.mem()->regGlobalTensor(cos);

        impl.forward(inputs, outputs, sin, cos);
      }

      break;
    }
    default: {
      NYI("Unsupported");
      break;
    }
  }
}

}  // namespace mllm::arm