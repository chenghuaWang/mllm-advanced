/**
 * @file SoftmaxOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/SoftmaxOp.hpp"

namespace mllm::arm {

class ArmSoftmaxOp final : public SoftmaxOp {
 public:
  explicit ArmSoftmaxOp(const SoftmaxOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmSoftmaxOpFactory : public TypedOpFactory<OpType::kSoftmax, SoftmaxOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SoftmaxOpCargo& cargo) override {
    return std::make_shared<ArmSoftmaxOp>(cargo);
  }
};

}  // namespace mllm::arm