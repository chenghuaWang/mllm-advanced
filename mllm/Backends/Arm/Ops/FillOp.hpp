/**
 * @file FillOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/FillOp.hpp"

namespace mllm::arm {

class ArmFillOp final : public FillOp {
 public:
  explicit ArmFillOp(const FillOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmFillOpFactory final : public TypedOpFactory<OpType::kFill, FillOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const FillOpCargo& cargo) override {
    return std::make_shared<ArmFillOp>(cargo);
  }
};

}  // namespace mllm::arm