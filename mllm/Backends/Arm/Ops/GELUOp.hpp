/**
 * @file GELUOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/GELUOp.hpp"

namespace mllm::arm {

class ArmGELUOp final : public GELUOp {
 public:
  explicit ArmGELUOp(const GELUOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmGELUOpFactory : public TypedOpFactory<OpType::kGELU, GELUOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const GELUOpCargo& cargo) override {
    return std::make_shared<ArmGELUOp>(cargo);
  }
};

}  // namespace mllm::arm