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
#include "mllm/Nn/Layers/Linear.hpp"
#include "mllm/Nn/Layers/SiLU.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Models/ds_qwen2/configuration_ds_qwen2.hpp"

namespace mllm::models {

class QWenMLP final : public nn::Module {
  nn::Linear gate_proj;
  nn::Linear up_proj;
  nn::Linear down_proj;
  nn::SiLU silu;

 public:
  QWenMLP(const QWenConfig& cfg) {
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

}  // namespace mllm::models
