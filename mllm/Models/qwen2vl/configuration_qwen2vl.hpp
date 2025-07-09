/**
 * @file configuration_qwen2vl.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Engine/CfgFile.hpp"

namespace mllm::models {

struct Qwen2VLConfig : protected MllmModelCfg {
  int32_t visual_in_chans = 3;
  int32_t visual_embed_dim = 1280;
  int32_t visual_patch_size = 14;
  int32_t visual_temporal_patch_size = 2;
  int32_t visual_spatial_merge_size = 2;
  int32_t visual_mlp_ratio = 4;
  int32_t visual_num_heads = 16;
  int32_t visual_depth = 32;

  int32_t hidden_size = 1536;
  int32_t intermediate_size = 8960;
  int32_t num_attention_heads = 12;
  int32_t num_key_value_heads = 2;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 32786;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 151936;

  LinearOpImplType linear_impl_type = LinearOpImplType::kDefault;
  DataTypes kv_cache_dtype = kFp32;
  int32_t max_cache_length = 2048;

  std::vector<int32_t> mrope_section = {16, 24, 24};

  float rope_theta = 1000000.0;
};

}  // namespace mllm::models
