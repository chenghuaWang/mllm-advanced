/**
 * @file modeling_qwen2vl.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/F/F.hpp"
#include "mllm/Nn/Layers/Conv3D.hpp"
#include "mllm/Nn/Layers/GELU.hpp"
#include "mllm/Nn/Layers/LayerNorm.hpp"
#include "mllm/Nn/Layers/Linear.hpp"
#include "mllm/Nn/Layers/SiLU.hpp"
#include "mllm/Nn/Layers/KVCache.hpp"
#include "mllm/Nn/Layers/CausalMask.hpp"
#include "mllm/Nn/Layers/Softmax.hpp"
#include "mllm/Nn/Layers/RMSNorm.hpp"
#include "mllm/Nn/Layers/VisionRoPE.hpp"
#include "mllm/Nn/Layers/MultimodalRoPE.hpp"
#include "mllm/Nn/Layers/LLMEmbeddingToken.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Core/Tensor.hpp"

#include "mllm/Models/qwen2vl/configuration_qwen2vl.hpp"

namespace mllm::models {

class PatchEmbed final : public nn::Module {
  int32_t in_chans_;
  int32_t embed_dim_;
  int32_t patch_size_;
  int32_t temporal_patch_size_;

  nn::Conv3D proj_;

 public:
  PatchEmbed() = default;

  inline explicit PatchEmbed(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);

    in_chans_ = cfg.visual_in_chans;
    embed_dim_ = cfg.visual_embed_dim;
    patch_size_ = cfg.visual_patch_size;
    temporal_patch_size_ = cfg.visual_temporal_patch_size;

    proj_ = reg<nn::Conv3D>("proj", cfg.visual_in_chans, cfg.visual_embed_dim,
                            std::vector<int32_t>{cfg.visual_temporal_patch_size,
                                                 cfg.visual_patch_size, cfg.visual_patch_size},
                            std::vector<int32_t>{cfg.visual_temporal_patch_size,
                                                 cfg.visual_patch_size, cfg.visual_patch_size},
                            false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto hidden_states = inputs[0];

    // [batch_size(1), in_channel(3), temporal_patch_size(2), patch_size(14), patch_size(14)]
    hidden_states =
        hidden_states.view({-1, in_chans_, temporal_patch_size_, patch_size_, patch_size_});
    hidden_states = proj_(hidden_states).view({-1, embed_dim_});

    return {hidden_states};
  }
};

class PatchMerger final : public nn::Module {
  int32_t hidden_size_;
  int32_t spatial_merge_size_;
  int32_t context_dim_;

  nn::LayerNorm ln_q_;
  nn::Linear mlp_0_;
  nn::Linear mlp_2_;
  nn::GELU mlp_gelu_;

 public:
  PatchMerger() = default;

  inline explicit PatchMerger(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);

    context_dim_ = cfg.visual_embed_dim;
    spatial_merge_size_ = cfg.visual_spatial_merge_size;
    hidden_size_ = context_dim_ * spatial_merge_size_ * spatial_merge_size_;

    ln_q_ = reg<nn::LayerNorm>("ln_q", context_dim_, true, true, 1e-6);
    mlp_0_ = reg<nn::Linear>("mlp.0", hidden_size_, hidden_size_, true);
    mlp_gelu_ = reg<nn::GELU>("mlp.gelu");
    mlp_2_ = reg<nn::Linear>("mlp.2", hidden_size_, cfg.hidden_size, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto o = ln_q_(inputs[0]).view({-1, hidden_size_});
    o = mlp_0_(o);
    o = mlp_gelu_(o);
    o = mlp_2_(o);
    return {o};
  }
};

class VisionMlp final : public nn::Module {
  int32_t dim_;
  int32_t hidden_dim_;

  nn::SiLU silu_;
  nn::Linear fc_1_;
  nn::Linear fc_2_;

 public:
  VisionMlp() = default;

  inline explicit VisionMlp(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);

    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;

    fc_1_ = reg<nn::Linear>("fc1", dim_, hidden_dim_, true);
    fc_2_ = reg<nn::Linear>("fc2", hidden_dim_, dim_, true);
    silu_ = reg<nn::SiLU>("silu");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    return {fc_2_(silu_(fc_1_(inputs[0])))};
  }
};

class VisionAttention final : public nn::Module {
  int32_t dim_;
  int32_t num_heads_;
  int32_t head_dim_;
  int32_t num_key_value_groups = 1;
  float scaling = 0.f;

  nn::Linear qkv_;
  nn::Linear proj_;
  nn::Softmax softmax_;
  nn::VisionRoPE vision_rope_q_;
  nn::VisionRoPE vision_rope_k_;

 public:
  VisionAttention() = default;

  inline explicit VisionAttention(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);

    dim_ = cfg.visual_embed_dim;
    num_heads_ = cfg.visual_num_heads;
    head_dim_ = dim_ / num_heads_;
    scaling = std::sqrt(head_dim_);

    qkv_ = reg<nn::Linear>("qkv", dim_, dim_ * 3, true);
    proj_ = reg<nn::Linear>("proj", dim_, dim_, true);
    softmax_ = reg<nn::Softmax>("softmax", -1);
    vision_rope_q_ = reg<nn::VisionRoPE>("vision_rope_q", VisionRoPEOpCargoType::kQwen2VL,
                                         Qwen2VLRoPEOpCargo{
                                             .dims = head_dim_,
                                             .spatial_merge_size = cfg.visual_spatial_merge_size,
                                             .theta = cfg.rope_theta,
                                         });
    vision_rope_k_ = reg<nn::VisionRoPE>("vision_rope_k", VisionRoPEOpCargoType::kQwen2VL,
                                         Qwen2VLRoPEOpCargo{
                                             .dims = head_dim_,
                                             .spatial_merge_size = cfg.visual_spatial_merge_size,
                                             .theta = cfg.rope_theta,
                                         });
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // hidden_states shape is [seq_length, dim]
    auto hidden_states = inputs[0];
    auto grid_thw = inputs[1];

    auto seq_length = hidden_states.shape()[0];

    auto [query_states, key_states, value_states] = nn::F::split<3>(
        qkv_(hidden_states).view({seq_length, 3, num_heads_, -1}).permute({1, 0, 2, 3}), 1, 0);

    // Input to Vision ROPE must be BSHD format
    // grid_thw shape is [n, 3], n is always 1 in this case.
    query_states = vision_rope_q_(query_states, grid_thw);
    key_states = vision_rope_k_(key_states, grid_thw);

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // attention weight
    // [B=1, H, S, S]
    auto attn = nn::F::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
    attn = softmax_(attn);
    // attn output
    // [B=1, H, S, S] @ [B=1, H, S, D] -> [B=1, H, S, D]
    auto attn_output = nn::F::matmul(attn, value_states);

    // [B=1, H, S, D] -> [B=1, S, H, D] -> [S, H * D]
    attn_output = attn_output.transpose(1, 2).view({seq_length, -1});
    attn_output = proj_(attn_output);
    return {attn_output};
  }
};

class Qwen2VLVisionBlock final : public nn::Module {
  int mlp_hidden_dim_;

  nn::LayerNorm norm1_;
  nn::LayerNorm norm2_;

  VisionAttention attn_;
  VisionMlp mlp_;

 public:
  Qwen2VLVisionBlock() = default;

  inline explicit Qwen2VLVisionBlock(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);

    mlp_hidden_dim_ = cfg.visual_mlp_ratio * cfg.visual_embed_dim;
    norm1_ = reg<nn::LayerNorm>("norm1", cfg.visual_embed_dim, true, true, 1e-6);
    norm2_ = reg<nn::LayerNorm>("norm2", cfg.visual_embed_dim, true, true, 1e-6);
    attn_ = reg<VisionAttention>("attn", cfg);
    mlp_ = reg<VisionMlp>("mlp", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto hidden_states = inputs[0];
    auto grid_thw = inputs[1];
    hidden_states = hidden_states + attn_(norm1_(hidden_states), grid_thw)[0];
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }
};

class Qwen2VisionTransformerPretrainedModel final : public nn::Module {
  PatchEmbed patch_embed_;
  PatchMerger patch_merger_;
  nn::ModuleList<Qwen2VLVisionBlock> blocks_;

 public:
  Qwen2VisionTransformerPretrainedModel() = default;

  explicit Qwen2VisionTransformerPretrainedModel(const std::string& name,
                                                 const Qwen2VLConfig& cfg) {
    selfAssignName(name);

    patch_embed_ = reg<PatchEmbed>("patch_embed", cfg);
    patch_merger_ = reg<PatchMerger>("merger", cfg);
    blocks_ = reg<nn::ModuleList<Qwen2VLVisionBlock>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto hidden_states = inputs[0];
    auto grid_thw = inputs[1];

    hidden_states = patch_embed_(hidden_states)[0];

    int cnt = 0;
    for (auto& b : blocks_.getList()) { hidden_states = b(hidden_states, grid_thw)[0]; }

    hidden_states = patch_merger_(hidden_states)[0];

    return {hidden_states};
  }
};

class Qwen2VLMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen2VLMLP() = default;
  explicit Qwen2VLMLP(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, false,
                                 cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, false,
                               cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, false,
                                 cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class Qwen2VLAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::MultimodalRoPE q_rope_;
  nn::MultimodalRoPE k_rope_;
  nn::KVCache k_cache_;
  nn::KVCache v_cache_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen2VLAttention() = default;

  Qwen2VLAttention(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true, false,
                              cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, false,
                              cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, false,
                              cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false,
                              false, cfg.linear_impl_type);

    q_rope_ = reg<nn::MultimodalRoPE>("q_rope", MultimodalRoPEOpCargoType::kQwen2VL, cfg.rope_theta,
                                      cfg.max_position_embeddings, cfg.mrope_section);
    k_rope_ = reg<nn::MultimodalRoPE>("k_rope", MultimodalRoPEOpCargoType::kQwen2VL, cfg.rope_theta,
                                      cfg.max_position_embeddings, cfg.mrope_section);

    k_cache_ = reg<nn::KVCache>("k_cache", num_key_value_heads_, head_dim_, num_key_value_groups_,
                                cfg.kv_cache_dtype, cfg.max_cache_length);
    v_cache_ = reg<nn::KVCache>("v_cache", num_key_value_heads_, head_dim_, num_key_value_groups_,
                                cfg.kv_cache_dtype, cfg.max_cache_length);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // [B, S, H * D]
    auto query_states = q_proj_(inputs[0]);
    auto key_states = k_proj_(inputs[1]);
    auto value_states = v_proj_(inputs[2]);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // [B, H, S, D]
    query_states = q_rope_(query_states);
    key_states = k_rope_(key_states);

    // [B, H, S, D]
    key_states = k_cache_(key_states);
    value_states = v_cache_(value_states);

    Tensor attn;
    if (key_states.dtype() == kFp32) {
      // attention weight
      // [B, H, S, S]
      attn = nn::F::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFp16) {
      attn = nn::F::matmul(query_states.to(kFp32), key_states.to(kFp32), false, true)
             * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      attn = attn.to(kFp16);
    }

    // attn output
    // [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    auto output = nn::F::matmul(attn, value_states);
    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);
    return {output};
  }
};

class Qwen2VLDecoder final : public nn::Module {
  Qwen2VLAttention self_attn_;
  Qwen2VLMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

 public:
  Qwen2VLDecoder() = default;

  Qwen2VLDecoder(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    self_attn_ = reg<Qwen2VLAttention>("self_attn", cfg);
    mlp_ = reg<Qwen2VLMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto x = input_layer_norm_(inputs[0]);
    x = self_attn_(x, x, x)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

class Qwen2VLText final : public nn::Module {
  nn::ModuleList<Qwen2VLDecoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen2VLText() = default;

  Qwen2VLText(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    decode_blocks_ = reg<nn::ModuleList<Qwen2VLDecoder>>("layers", cfg.num_hidden_layers, cfg);
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::LLMEmbeddingToken>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto& blocks = decode_blocks_.getList();
    auto x = inputs[0];
    for (auto& block : blocks) { x = block(x)[0]; }
    x = norm_(x);

    auto lm_head = params("embed_tokens.weight");

    // x is [B, S, D], lm_head is [V, D]
    x = nn::F::matmul(x, lm_head, false, true);

    return {x};
  }

  nn::LLMEmbeddingToken embedding_;
};

class Qwen2VLForCausalLM {
 public:
  explicit Qwen2VLForCausalLM(const Qwen2VLConfig& cfg)
      : llm("model", cfg), visual("visual", cfg) {}

  inline Qwen2VLForCausalLMOutputPast operator()(const Qwen2VLForCausalLMOutputPast& past) {
    // Calculate the text embeddings
    auto input_embeddings = llm.embedding_(past.sequence);

    if (!past.img.isNil()) {
      // process img
      auto visual_embeddings = visual(past.img, past.grid_thw)[0];

      // Insert visual embeddings into llm's embedding
      // TODO
    }

    if (past.position_ids.isNil()) {
      // TODO
    }

    auto out = llm(input_embeddings)[0];

    return {
        .sequence = out,
        .img = Tensor::nil(),
        .grid_thw = past.grid_thw,
    };
  }

  inline void getPositionIds(const Qwen2VLForCausalLMOutputPast& past) {
    // Input is [B, S, D]
    if (!past.img.isNil()) {  // Prefill
      // TODO
    } else {  // Decode
      // TODO
    }
  }

  inline Tensor getPositionIdsPrefill(Tensor& input_ids, Tensor& image_grid_thw) {
    // Input is [B, S, D]
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 3);
    // image_grid_thw is [num_images, 3]
    MLLM_RT_ASSERT_EQ(image_grid_thw.shape().size(), 2);

    auto B = input_ids.shape()[0];
    MLLM_RT_ASSERT_EQ(B, 1);
    auto S = input_ids.shape()[1];
    auto D = input_ids.shape()[2];

    Tensor position_ids = Tensor::empty({3, B, S}, kFp32, kCPU).alloc();

    // Process text and visual
    // TODO
    // https://github.com/UbiquitousLearning/mllm/blob/main/src/models/qwen2_vl/modeling_qwen2_vl.hpp

    return position_ids;
  }

  Qwen2VLText llm;
  Qwen2VisionTransformerPretrainedModel visual;
};

}  // namespace mllm::models
