/**
 * @file SiLUOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/SiLUOp.hpp"

namespace mllm::arm {

class ArmSiLUOp final : public SiLUOp {
 public:
  explicit ArmSiLUOp(const SiLUOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmSiLUOpFactory : public TypedOpFactory<OpType::kTranspose, SiLUOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SiLUOpCargo& cargo) override {
    return std::make_shared<ArmSiLUOp>(cargo);
  }
};

}  // namespace mllm::arm