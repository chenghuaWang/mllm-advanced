/**
 * @file RepeatOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/RepeatOp.hpp"

namespace mllm::arm {

class ArmRepeatOp final : public RepeatOp {
 public:
  explicit ArmRepeatOp(const RepeatOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmRepeatOpFactory : public TypedOpFactory<OpType::kRepeat, RepeatOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const RepeatOpCargo& cargo) override {
    return std::make_shared<ArmRepeatOp>(cargo);
  }
};

}  // namespace mllm::arm