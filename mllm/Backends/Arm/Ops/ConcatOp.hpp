/**
 * @file ConcatOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/ConcatOp.hpp"

namespace mllm::arm {

class ArmConcatOp final : public ConcatOp {
 public:
  explicit ArmConcatOp(const ConcatOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmConcatOpFactory : public TypedOpFactory<OpType::kConcat, ConcatOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const ConcatOpCargo& cargo) override {
    return std::make_shared<ArmConcatOp>(cargo);
  }
};

}  // namespace mllm::arm