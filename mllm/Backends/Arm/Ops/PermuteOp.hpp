/**
 * @file PermuteOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/PermuteOp.hpp"

namespace mllm::arm {

class ArmPermuteOp final : public PermuteOp {
 public:
  explicit ArmPermuteOp(const PermuteOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmPermuteOpFactory : public TypedOpFactory<OpType::kPermute, PermuteOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const PermuteOpCargo& cargo) override {
    return std::make_shared<ArmPermuteOp>(cargo);
  }
};

}  // namespace mllm::arm