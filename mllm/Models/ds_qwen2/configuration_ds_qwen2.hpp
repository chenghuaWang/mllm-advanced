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

#include <cstdint>
#include <string>
#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Engine/CfgFile.hpp"

namespace mllm::models {

struct QWenConfig : protected MllmModelCfg {
  explicit QWenConfig(const std::string& cfg_file_path) : MllmModelCfg(cfg_file_path) {
    num_attention_heads = json_["num_attention_heads"].get<int>();
    num_hidden_layers = json_["num_hidden_layers"].get<int>();
    num_key_value_heads = json_["num_key_value_heads"].get<int>();
    hidden_size = json_["hidden_size"].get<int>();
    intermediate_size = json_["intermediate_size"].get<int>();
    rope_theta = json_["rope_theta"].get<float>();
    max_position_embeddings = json_["max_position_embeddings"].get<int>();
    rms_norm_eps = json_["rms_norm_eps"].get<float>();
    vocab_size = json_["vocab_size"].get<int>();
    max_cache_length = json_["max_cache_length"].get<int>();

    eos_token_id = json_["eos_token_id"].get<int64_t>();

    auto kv_cache_dtype_str = json_["kv_cache_dtype"].get<std::string>();
    if (kv_cache_dtype_str == "Fp32") {
      kv_cache_dtype = DataTypes::kFp32;
    } else if (kv_cache_dtype_str == "Fp16") {
      kv_cache_dtype = DataTypes::kFp16;
    }

    auto linear_impl_type_str = json_["linear_impl_type"].get<std::string>();
    linear_impl_type = LinearOpCargo::parseLinearOpImplTypeStr(linear_impl_type_str);
  }

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

  DataTypes kv_cache_dtype = kFp32;
  LinearOpImplType linear_impl_type = LinearOpImplType::kDefault;

  int64_t eos_token_id = 151643;
};
}  // namespace mllm::models
