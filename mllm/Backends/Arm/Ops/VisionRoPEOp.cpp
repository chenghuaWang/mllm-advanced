/**
 * @file VisionRoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include "mllm/Backends/Arm/Ops/VisionRoPEOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

Tensor Qwen2VLVisionRoPEOpImpl::computeInvFreq(const Qwen2VLRoPEOpCargo& cargo) {
  const int half_dim = cargo.dims / (2 * 2);
  Tensor inv_freq = Tensor::empty({half_dim}, kFp32).alloc();
  float* inv_freq_ptr = inv_freq.ptr<float>();

  const float theta = cargo.theta;
  const float dims_inv = 1.0f / static_cast<float>(cargo.dims / 2);

  for (int i = 0; i < half_dim; ++i) {
    const float exponent = (2.0f * i) * dims_inv;
    inv_freq_ptr[i] = 1.0f / std::pow(theta, exponent);
  }

  return inv_freq;
}

Tensor Qwen2VLVisionRoPEOpImpl::getRotaryPosEmbIds(Tensor& grid_thw,
                                                   const Qwen2VLRoPEOpCargo& cargo) {
  MLLM_RT_ASSERT_EQ(grid_thw.shape().size(), 2);

  auto img_nums = grid_thw.shape()[0];
  const int spatial_merge_size = cargo.spatial_merge_size;

  int total_positions = 0;
  for (int row = 0; row < img_nums; ++row) {
    const int* dims = grid_thw.offsettedPtr<int>({row, 0});
    const int t = dims[0];
    const int h = dims[1];
    const int w = dims[2];
    total_positions += t * h * w;
  }

  Tensor out = Tensor::empty({total_positions, 2}, kInt32).alloc();
  int* out_ptr = out.ptr<int>();
  int out_offset = 0;

  const int32_t neon_steps_arr[4] = {0, 1, 2, 3};
  const int32x4_t neon_steps = vld1q_s32(neon_steps_arr);

  for (int row = 0; row < img_nums; ++row) {
    const int* dims = grid_thw.offsettedPtr<int>({row, 0});

    const int t = dims[0];
    const int h = dims[1];
    const int w = dims[2];

    const int num_h_blocks = h / spatial_merge_size;
    const int num_w_blocks = w / spatial_merge_size;
    const int total_blocks = num_h_blocks * num_w_blocks;
    const int block_area = spatial_merge_size * spatial_merge_size;
    const int grid_size = h * w;

    std::vector<int> flatten_hpos(grid_size);
    std::vector<int> flatten_wpos(grid_size);

    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
      const int i_h = block_idx / num_w_blocks;
      const int i_w = block_idx % num_w_blocks;
      const int start_idx = block_idx * block_area;

      const int base_h = i_h * spatial_merge_size;
      const int base_w = i_w * spatial_merge_size;

      for (int j_h = 0; j_h < spatial_merge_size; ++j_h) {
        const int global_h = base_h + j_h;
        const int row_start = start_idx + j_h * spatial_merge_size;

        int j_w = 0;
        for (; j_w <= spatial_merge_size - 4; j_w += 4) {
          const int32x4_t jw_offset = vaddq_s32(vdupq_n_s32(j_w), neon_steps);
          const int32x4_t global_w = vaddq_s32(vdupq_n_s32(base_w), jw_offset);
          const int pos = row_start + j_w;
          vst1q_s32(&flatten_wpos[pos], global_w);
          vst1_s32(&flatten_hpos[pos], vdup_n_s32(global_h));
        }
        for (; j_w < spatial_merge_size; ++j_w) {
          const int pos = row_start + j_w;
          flatten_hpos[pos] = global_h;
          flatten_wpos[pos] = base_w + j_w;
        }
      }
    }

    for (int frame = 0; frame < t; ++frame) {
      const int frame_offset = frame * grid_size * 2;
      for (int pos = 0; pos < grid_size; ++pos) {
        const int out_idx = out_offset + frame_offset + pos * 2;
        out_ptr[out_idx] = flatten_hpos[pos];
        out_ptr[out_idx + 1] = flatten_wpos[pos];
      }
    }
    out_offset += t * grid_size * 2;
  }

  return out;
}

Tensor Qwen2VLVisionRoPEOpImpl::computeRotaryPosEmb(Tensor& rotary_pos_emb_full, Tensor& pos_ids,
                                                    Tensor& grid_thw,
                                                    const Qwen2VLRoPEOpCargo& cargo) {
  const int* grid_dims = grid_thw.offsettedPtr<int>({0, 0});
  const int t = grid_dims[0];
  const int h = grid_dims[1];
  const int w = grid_dims[2];

  const int32_t num_positions = rotary_pos_emb_full.shape()[0];
  const int32_t dim = rotary_pos_emb_full.shape()[1];
  const int32_t batch_size = pos_ids.shape()[0];
  const int32_t seq_len = pos_ids.shape()[1];

  // [batch_size, dim]
  Tensor out = Tensor::empty({batch_size, seq_len * dim}, kFp32, kCPU).alloc();

  auto rotary_pos_emb_full_ptr = rotary_pos_emb_full.ptr<float>();
  auto pos_ids_ptr = pos_ids.ptr<int>();
  auto out_ptr = out.ptr<float>();

  if (num_positions <= 0 || dim <= 0 || batch_size <= 0) {
    MLLM_ERROR_EXIT(kError, "Invalid tensor dimensions");
  }

  if (t * h * w != batch_size) {
    MLLM_ERROR_EXIT(kError, "Grid dimensions mismatch with batch size");
  }

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      if ((*pos_ids.offsettedPtr<int>({i, j})) < 0
          || (*pos_ids.offsettedPtr<int>({i, j})) >= num_positions) {
        MLLM_ERROR_EXIT(kError, "Position index out of bounds");
      }
    }
  }

  for (int i = 0; i < batch_size; ++i) {
    auto batch_ptr = out.offsettedPtr<float>({i, 0});
    size_t offset = 0;
    for (int j = 0; j < seq_len; ++j) {
      auto emb_ptr =
          rotary_pos_emb_full.offsettedPtr<float>({(*pos_ids.offsettedPtr<int>({i, j})), 0});
      std::copy(emb_ptr, emb_ptr + dim, batch_ptr + offset);
      offset += dim;
    }
  }

  return out;
}

Tensor Qwen2VLVisionRoPEOpImpl::rotaryPosEmb(Tensor& inv_freq, int seq_len,
                                             const Qwen2VLRoPEOpCargo& cargo) {
  MLLM_RT_ASSERT(seq_len > 0);
  const int32_t dim = inv_freq.shape()[0];
  Tensor freqs = Tensor::empty({seq_len, dim}, kFp32, kCPU).alloc();

  float* inv_freq_ptr = inv_freq.ptr<float>();
  float* freqs_ptr = freqs.ptr<float>();

  for (int i = 0; i < seq_len; ++i) {
    const float i_val = static_cast<float>(i);
    float* row_ptr = freqs_ptr + i * dim;

    size_t j = 0;
    const float32x4_t v_i = vdupq_n_f32(i_val);

    for (; j + 3 < dim; j += 4) {
      const float32x4_t v_inv = vld1q_f32(inv_freq_ptr + j);
      const float32x4_t v_res = vmulq_f32(v_i, v_inv);
      vst1q_f32(row_ptr + j, v_res);
    }

    for (; j < dim; ++j) { row_ptr[j] = i_val * inv_freq_ptr[j]; }
  }

  return freqs;
}

std::pair<Tensor, Tensor> Qwen2VLVisionRoPEOpImpl::getSinCos(Tensor& rotary_pos_emb) {
  auto seq = rotary_pos_emb.shape()[0];
  auto dim = rotary_pos_emb.shape()[1];

  auto rotary_pos_emb_ptr = rotary_pos_emb.ptr<float>();

  Tensor sin_pos_emb = Tensor::empty({seq, dim}, kFp32, kCPU).alloc();
  Tensor cos_pos_emb = Tensor::empty({seq, dim}, kFp32, kCPU).alloc();

  auto sin_pos_emb_ptr = sin_pos_emb.ptr<float>();
  auto cos_pos_emb_ptr = cos_pos_emb.ptr<float>();

  for (int i = 0; i < seq; i++) {
    for (int j = 0; j < dim; j++) {
      sin_pos_emb_ptr[i * dim + j] = std::sin(rotary_pos_emb_ptr[i * dim + j]);
      cos_pos_emb_ptr[i * dim + j] = std::cos(rotary_pos_emb_ptr[i * dim + j]);
    }
  }

  return {sin_pos_emb, cos_pos_emb};
}

void Qwen2VLVisionRoPEOpImpl::forward(const Tensor& activation, const Tensor& sin,
                                      const Tensor& cos, Tensor& out) {
  // [B, S, H, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  switch (activation.dtype()) {
    case kFp16: {
      auto B = activation.shape()[0];
      auto S = activation.shape()[1];
      auto H = activation.shape()[2];
      auto D = activation.shape()[3];
      auto activation_ptr = activation.ptr<float16_t>();
      auto output_ptr = out.ptr<float16_t>();
      auto sin_ptr = sin.ptr<float16_t>();
      auto cos_ptr = cos.ptr<float16_t>();

      auto half_dim = D / 2;

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            auto act_base = activation_ptr + b * S * H * D + s * H * D + h * D;
            auto out_base = output_ptr + b * S * H * D + s * H * D + h * D;

            auto sin_base = sin_ptr + s * half_dim;  // sin shape is [S, half_dim]
            auto cos_base = cos_ptr + s * half_dim;  // cos shape is [S, half_dim]

            int d = 0;
            for (; d + 3 < half_dim; d += 4) {
              float16x4_t a = vld1_f16(act_base + d);
              float16x4_t b = vld1_f16(act_base + d + half_dim);

              float16x4_t cos_val = vld1_f16(cos_base + d);
              float16x4_t sin_val = vld1_f16(sin_base + d);

              // part1 = a * cos_val - b * sin_val
              // part2 = a * sin_val + b * cos_val
              float16x4_t part1 = vsub_f16(vmul_f16(a, cos_val), vmul_f16(b, sin_val));
              float16x4_t part2 = vadd_f16(vmul_f16(a, sin_val), vmul_f16(b, cos_val));

              vst1_f16(out_base + d, part1);
              vst1_f16(out_base + d + half_dim, part2);
            }

            for (; d < half_dim; ++d) {
              const auto a = act_base[d];
              const auto b = act_base[d + half_dim];
              const auto cos_val = cos_base[d];
              const auto sin_val = sin_base[d];
              out_base[d] = a * cos_val - b * sin_val;
              out_base[d + half_dim] = a * sin_val + b * cos_val;
            }
          }
        }
      }
      break;
    }
    case kFp32: {
      const auto B = activation.shape()[0];
      const auto S = activation.shape()[1];
      const auto H = activation.shape()[2];
      const auto D = activation.shape()[3];
      const auto half_dim = D / 2;

      auto activation_ptr = activation.ptr<float>();
      auto output_ptr = out.ptr<float>();
      auto sin_ptr = sin.ptr<float>();
      auto cos_ptr = cos.ptr<float>();

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            auto act_base = activation_ptr + b * S * H * D + s * H * D + h * D;
            auto out_base = output_ptr + b * S * H * D + s * H * D + h * D;

            auto sin_base = sin_ptr + s * half_dim;  // sin shape is [S, half_dim]
            auto cos_base = cos_ptr + s * half_dim;  // cos shape is [S, half_dim]

            int d = 0;
            for (; d + 3 < half_dim; d += 4) {
              float32x4_t a_front = vld1q_f32(act_base + d);
              float32x4_t a_back = vld1q_f32(act_base + d + half_dim);

              float32x4_t cos_val = vld1q_f32(cos_base + d);
              float32x4_t sin_val = vld1q_f32(sin_base + d);

              // out_front = a_front * cos_val - a_back * sin_val
              // out_back  = a_front * sin_val + a_back * cos_val
              float32x4_t out_front = vmlsq_f32(vmulq_f32(a_front, cos_val), a_back, sin_val);
              float32x4_t out_back = vmlaq_f32(vmulq_f32(a_front, sin_val), a_back, cos_val);

              vst1q_f32(out_base + d, out_front);
              vst1q_f32(out_base + d + half_dim, out_back);
            }

            for (; d < half_dim; ++d) {
              const float a_front = act_base[d];
              const float a_back = act_base[d + half_dim];
              const float cos_val = cos_base[d];
              const float sin_val = sin_base[d];
              out_base[d] = a_front * cos_val - a_back * sin_val;
              out_base[d + half_dim] = a_front * sin_val + a_back * cos_val;
            }
          }
        }
      }
      break;
    }
    default: {
      NYI("Unsupported activation type");
    }
  }
}

ArmVisionRoPEOp::ArmVisionRoPEOp(const VisionRoPEOpCargo& cargo) : VisionRoPEOp(cargo) {}

void ArmVisionRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& ctx = MllmEngineCtx::instance();

  auto& activation = inputs[0];
  auto grid_thw = inputs[1];

  // Only support BSHD inputs.
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  // Input is BSHD, seq_len is in pos 1.
  auto seq_len = activation.shape()[1];

  switch (cargo_.type) {
    case VisionRoPEOpCargoType::kQwen2VL: {
      // For Qwen2VL's ViT RoPE
      auto impl = Qwen2VLVisionRoPEOpImpl();

      Tensor sin = Tensor::nil();
      Tensor cos = Tensor::nil();

      // Gen/load sin and cos.
      if (ctx.mem()->hasGlobalTensor("__qwen2vl_vit_rope_sin")
          && ctx.mem()->hasGlobalTensor("__qwen2vl_vit_rope_cos")) {
        sin = ctx.mem()->getGlobalTensor("__qwen2vl_vit_rope_sin");
        cos = ctx.mem()->getGlobalTensor("__qwen2vl_vit_rope_cos");
      } else {
        auto max_grid_size = -1;
        {
          auto grid_ptr = grid_thw.ptr<int>();
          max_grid_size = std::max({grid_ptr[0], grid_ptr[1], grid_ptr[2]});
        }
        auto inv_freq = impl.computeInvFreq(cargo_.qwen2vl_rope_op_cargo);
        auto pos_ids = impl.getRotaryPosEmbIds(grid_thw, cargo_.qwen2vl_rope_op_cargo);
        auto rotary_pos_emb_full =
            impl.rotaryPosEmb(inv_freq, max_grid_size, cargo_.qwen2vl_rope_op_cargo);
        auto pos_emb = impl.computeRotaryPosEmb(rotary_pos_emb_full, pos_ids, grid_thw,
                                                cargo_.qwen2vl_rope_op_cargo);

        auto sin_cos = impl.getSinCos(pos_emb);

        sin = sin_cos.first;
        cos = sin_cos.second;

        sin = sin.to(activation.dtype());
        cos = cos.to(activation.dtype());

        sin = sin.setMemType(kGlobal).setName("__qwen2vl_vit_rope_sin");
        cos = cos.setMemType(kGlobal).setName("__qwen2vl_vit_rope_cos");

        ctx.mem()->regGlobalTensor(sin);
        ctx.mem()->regGlobalTensor(cos);
      }

      impl.forward(activation, sin, cos, outputs[0]);
      break;
    }
    default: {
      NYI("VisionRoPEOpCargoType is not impl");
      break;
    }
  }
}

}  // namespace mllm::arm