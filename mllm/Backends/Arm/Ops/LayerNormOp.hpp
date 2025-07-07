/**
 * @file LayerNormOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/LayerNormOp.hpp"

namespace mllm::arm {

class ArmLayerNormOp final : public LayerNormOp {
 public:
  explicit ArmLayerNormOp(const LayerNormOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmLayerNormOpFactory : public TypedOpFactory<OpType::kLayerNorm, LayerNormOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const LayerNormOpCargo& cargo) override {
    return std::make_shared<ArmLayerNormOp>(cargo);
  }
};

}  // namespace mllm::arm