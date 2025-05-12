/**
 * @file configuration_ds_qwen2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>
#include <cstdint>
#include "mllm/Core/DataTypes.hpp"

namespace mllm::models {
struct QWenConfig {
  std::string gate_proj_name = "gate_proj";
  std::string up_proj_name = "up_proj";
  std::string down_proj_name = "down_proj";
  std::string q_proj_name = "q_proj";
  std::string k_proj_name = "k_proj";
  std::string v_proj_name = "v_proj";
  std::string o_proj_name = "o_proj";
  std::string q_rope_name = "q_rope";
  std::string k_rope_name = "k_rope";
  std::string k_cache_name = "k_cache";
  std::string v_cache_name = "v_cache";
  std::string mask_name = "mask";
  std::string softmax_name = "softmax";

  std::string attn_base_name = "self_attn";
  std::string ffn_base_name = "mlp";
  std::string attn_norm_name = "input_layernorm";
  std::string ffn_norm_name = "post_attention_layernorm";

  std::string post_norm_name = "norm";

  std::string top_module_name = "model";
  std::string layers_base_name = "layers";
  std::string lm_head_weight_name = "lm_head.weight";
  std::string emb_token_name = "embed_tokens";

  int num_attention_heads = 12;
  int num_hidden_layers = 28;
  int num_key_value_heads = 2;
  int hidden_size = 1536;
  int intermediate_size = 8960;
  float rope_theta = 10000.f;
  int max_position_embeddings = 131072;
  float rms_norm_eps = 1e-06;
  int vocab_size = 151936;

  int max_cache_length = 1024;

  DataTypes kv_cache_dtype = kFp16;

  int64_t eos_token_id = 151643;
};
}  // namespace mllm::models
