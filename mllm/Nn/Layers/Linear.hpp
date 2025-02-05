/**
 * @file Linear.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/LinearOp.hpp"

namespace mllm::nn {

class Linear : public Layer {
 public:
  Linear();

  Linear(int32_t in_channels, int32_t out_channels, bool bias = true, bool transpose = false);

  explicit Linear(const LinearOpCargo& cargo);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;
};

}  // namespace mllm::nn
