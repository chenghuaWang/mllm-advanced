/**
 * @file QuickGELUOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/QuickGELUOp.hpp"

namespace mllm::arm {

class ArmQuickGELUOp final : public QuickGELUOp {
 public:
  explicit ArmQuickGELUOp(const QuickGELUOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmQuickGELUOpFactory : public TypedOpFactory<OpType::kQuickGELU, QuickGELUOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const QuickGELUOpCargo& cargo) override {
    return std::make_shared<ArmQuickGELUOp>(cargo);
  }
};

}  // namespace mllm::arm