/**
 * @file ViewOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/ViewOp.hpp"

namespace mllm::arm {

class ArmViewOp final : public ViewOp {
 public:
  explicit ArmViewOp(const ViewOpCargo& cargo);
};

class ArmViewOpFactory : public TypedOpFactory<OpType::kView, ViewOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const ViewOpCargo& cargo) override {
    return std::make_shared<ArmViewOp>(cargo);
  }
};

}  // namespace mllm::arm