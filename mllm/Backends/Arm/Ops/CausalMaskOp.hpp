/**
 * @file CausalMaskOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/CausalMaskOp.hpp"

namespace mllm::arm {

class ArmCausalMaskOp final : public CausalMaskOp {
 public:
  explicit ArmCausalMaskOp(const CausalMaskOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmCausalMaskOpFactory : public TypedOpFactory<OpType::kCausalMask, CausalMaskOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const CausalMaskOpCargo& cargo) override {
    return std::make_shared<ArmCausalMaskOp>(cargo);
  }
};

}  // namespace mllm::arm