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

#include "mllm/Core/Tensor.hpp"
#include "mllm/Nn/Layers/KVCache.hpp"
#include "mllm/Nn/Layers/Linear.hpp"
#include "mllm/Nn/Layers/SiLU.hpp"
#include "mllm/Nn/Layers/Softmax.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Models/ds_qwen2/configuration_ds_qwen2.hpp"

namespace mllm::models {

class QWenMLP final : public nn::Module {
  nn::Linear gate_proj;
  nn::Linear up_proj;
  nn::Linear down_proj;
  nn::SiLU silu;

 public:
  QWenMLP() = default;
  explicit QWenMLP(const QWenConfig& cfg) {
    gate_proj = reg<nn::Linear>(cfg.gate_proj_name, cfg.hidden_size, cfg.intermediate_size, false);
    silu = reg<nn::SiLU>("act");
    up_proj = reg<nn::Linear>(cfg.up_proj_name, cfg.hidden_size, cfg.intermediate_size, false);
    down_proj = reg<nn::Linear>(cfg.down_proj_name, cfg.intermediate_size, cfg.hidden_size, false);
  }

  std::vector<Tensor> forward(std::vector<Tensor>& inputs) override {
    auto x = gate_proj(inputs[0]);
    x = silu(x);
    auto y = up_proj(inputs[0]);
    x = x * y;
    x = down_proj(x);
    return {x};
  }
};

class QWenAttention final : public nn::Module {
  nn::Linear q_proj;
  nn::Linear k_proj;
  nn::Linear v_proj;
  nn::Linear o_proj;
  // nn::RoPE q_rope;
  // nn::RoPE k_rope;
  nn::KVCache k_cache;
  nn::KVCache v_cache;
  // nn::Causalmask mask;
  nn::Softmax softmax;

 public:
  QWenAttention() = default;

  QWenAttention(const QWenConfig& config) {}

  std::vector<Tensor> forward(std::vector<Tensor>& inputs) override {
    // [B, S, H * D]
    auto query_states = q_proj(inputs[0]);
    auto key_states = k_proj(inputs[1]);
    auto value_states = v_proj(inputs[2]);

    // [B, S, H, D]
    query_states = query_states.view({});
    key_states = key_states.view({});
    value_states = value_states.view({});

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // RoPE
    // TODO

    // [B, H, S, D]
    key_states = k_cache(key_states);
    value_states = v_cache(value_states);

    // matmul TODO
  }
};

}  // namespace mllm::models
