/**
 * @file LinearOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/LinearOp.hpp"

namespace mllm::arm {

class ArmLinearOp final : public LinearOp {
 public:
  explicit ArmLinearOp(const LinearOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmLinearOpFactory : public TypedOpFactory<OpType::kLinear, LinearOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const LinearOpCargo& cargo) override {
    return std::make_shared<ArmLinearOp>(cargo);
  }
};

}  // namespace mllm::arm