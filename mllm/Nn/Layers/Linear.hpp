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
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/LinearOp.hpp"

namespace mllm::nn {

class Linear : public Layer {
 public:
  Linear() = default;
  explicit Linear(LinearOpCargo cargo);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bisa() const;

 private:
  OpType op_type_ = OpType::kLinear;
  LinearOpCargo cargo_;
};

}  // namespace mllm::nn
