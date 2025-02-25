/**
 * @file CastTypeOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/CastTypeOp.hpp"

namespace mllm::arm {

class ArmCastTypeOp final : public CastTypeOp {
 public:
  explicit ArmCastTypeOp(const CastTypeOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmCastTypeOpFactory : public TypedOpFactory<OpType::kCastType, CastTypeOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const CastTypeOpCargo& cargo) override {
    return std::make_shared<ArmCastTypeOp>(cargo);
  }
};

}  // namespace mllm::arm