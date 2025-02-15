/**
 * @file RMSNorm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/RMSNormOp.hpp"

namespace mllm::arm {

class ArmRMSNormOp final : public RMSNormOp {
 public:
  explicit ArmRMSNormOp(const RMSNormOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmRMSNormOpFactory : public TypedOpFactory<OpType::kRMSNorm, RMSNormOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const RMSNormOpCargo& cargo) override {
    return std::make_shared<ArmRMSNormOp>(cargo);
  }
};

}  // namespace mllm::arm
