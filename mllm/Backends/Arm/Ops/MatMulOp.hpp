/**
 * @file MatMulOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/MatMulOp.hpp"

namespace mllm::arm {

class ArmMatMulOp final : public MatMulOp {
 public:
  explicit ArmMatMulOp(const MatMulOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmMatMulOpFactory : public TypedOpFactory<OpType::kMatMul, MatMulOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const MatMulOpCargo& cargo) override {
    return std::make_shared<ArmMatMulOp>(cargo);
  }
};

}  // namespace mllm::arm
