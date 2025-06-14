/**
 * @file modeling_ds_qwen2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cmath>
#include "mllm/Core/Tensor.hpp"
#include "mllm/Nn/F/F.hpp"
#include "mllm/Nn/Layers/CausalMask.hpp"
#include "mllm/Nn/Layers/KVCache.hpp"
#include "mllm/Nn/Layers/LLMEmbeddingToken.hpp"
#include "mllm/Nn/Layers/Linear.hpp"
#include "mllm/Nn/Layers/RMSNorm.hpp"
#include "mllm/Nn/Layers/RoPE.hpp"
#include "mllm/Nn/Layers/SiLU.hpp"
#include "mllm/Nn/Layers/Softmax.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Models/ds_qwen2/configuration_ds_qwen2.hpp"

namespace mllm::models {

class QWenMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  QWenMLP() = default;
  explicit QWenMLP(const std::string& name, const QWenConfig& cfg) {
    selfAssignName(name);
    gate_proj_ = reg<nn::Linear>(cfg.gate_proj_name, cfg.hidden_size, cfg.intermediate_size, false,
                                 false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>(cfg.up_proj_name, cfg.hidden_size, cfg.intermediate_size, false,
                               false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>(cfg.down_proj_name, cfg.intermediate_size, cfg.hidden_size, false,
                                 false, cfg.linear_impl_type);
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

class QWenAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
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
  QWenAttention() = default;

  QWenAttention(const std::string& name, const QWenConfig& cfg) {
    selfAssignName(name);
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>(cfg.q_proj_name, hidden_size_, head_dim_ * num_attention_heads_, true,
                              false, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>(cfg.k_proj_name, hidden_size_, head_dim_ * num_key_value_heads_, true,
                              false, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>(cfg.v_proj_name, hidden_size_, head_dim_ * num_key_value_heads_, true,
                              false, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>(cfg.o_proj_name, head_dim_ * num_attention_heads_, hidden_size_,
                              false, false, cfg.linear_impl_type);
    q_rope_ = reg<nn::RoPE>(cfg.q_rope_name, RoPETypes::kLlama2, cfg.rope_theta,
                            cfg.max_position_embeddings, head_dim_);
    k_rope_ = reg<nn::RoPE>(cfg.k_rope_name, RoPETypes::kLlama2, cfg.rope_theta,
                            cfg.max_position_embeddings, head_dim_);
    k_cache_ = reg<nn::KVCache>(cfg.k_cache_name, num_key_value_heads_, head_dim_,
                                num_key_value_groups_, cfg.kv_cache_dtype, cfg.max_cache_length);
    v_cache_ = reg<nn::KVCache>(cfg.v_cache_name, num_key_value_heads_, head_dim_,
                                num_key_value_groups_, cfg.kv_cache_dtype, cfg.max_cache_length);
    mask_ = reg<nn::CausalMask>(cfg.mask_name);
    softmax_ = reg<nn::Softmax>(cfg.softmax_name, -1);
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

class QWenDecoder final : public nn::Module {
  QWenAttention self_attn_;
  QWenMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

 public:
  QWenDecoder() = default;

  QWenDecoder(const std::string& name, const QWenConfig& cfg) {
    selfAssignName(name);
    self_attn_ = reg<QWenAttention>(cfg.attn_base_name, cfg);
    mlp_ = reg<QWenMLP>(cfg.ffn_base_name, cfg);
    input_layer_norm_ = reg<nn::RMSNorm>(cfg.attn_norm_name, cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>(cfg.ffn_norm_name, cfg.rms_norm_eps);
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

class QWenForCausalLM final : public nn::Module {
  nn::ModuleList<QWenDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::LLMEmbeddingToken embedding_;

 public:
  QWenForCausalLM() = default;

  explicit QWenForCausalLM(const QWenConfig& cfg) {
    selfAssignName(cfg.top_module_name);
    decode_blocks_ =
        reg<nn::ModuleList<QWenDecoder>>(cfg.layers_base_name, cfg.num_hidden_layers, cfg);
    norm_ = reg<nn::RMSNorm>(cfg.post_norm_name, cfg.rms_norm_eps);
    embedding_ = reg<nn::LLMEmbeddingToken>(cfg.emb_token_name, cfg.vocab_size, cfg.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto& blocks = decode_blocks_.getList();
    auto x = inputs[0];
    x = embedding_(x);
    for (auto& block : blocks) { x = block(x)[0]; }
    x = norm_(x);

    auto lm_head = params("lm_head.weight");

    // x is [B, S, D], lm_head is [V, D]
    x = nn::F::matmul(x, lm_head, false, true);

    return {x};
  }
};

}  // namespace mllm::models
