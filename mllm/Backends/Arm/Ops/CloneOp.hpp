/**
 * @file CloneOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/CloneOp.hpp"

namespace mllm::arm {

class ArmCloneOp final : public CloneOp {
 public:
  explicit ArmCloneOp(const CloneOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmCloneOpFactory : public TypedOpFactory<OpType::kClone, CloneOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const CloneOpCargo& cargo) override {
    return std::make_shared<ArmCloneOp>(cargo);
  }
};

}  // namespace mllm::arm