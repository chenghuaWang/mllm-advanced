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

#include "mllm/Nn/Module.hpp"

namespace mllm::models {

class DecodingLayer final : public nn::Module {
 public:
  std::vector<Tensor> forward(std::vector<Tensor>& inputs) override { return {}; }

 private:
};

}  // namespace mllm::models
