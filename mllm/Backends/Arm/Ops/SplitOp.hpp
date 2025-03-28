/**
 * @file SplitOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/SplitOp.hpp"

namespace mllm {

class ArmSplitOp final : public SplitOp {
 public:
  explicit ArmSplitOp(const SplitOpCargo& cargo);
};

class ArmSplitOpFactory : public TypedOpFactory<OpType::kSplit, SplitOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SplitOpCargo& cargo) override {
    return std::make_shared<ArmSplitOp>(cargo);
  }
};

}  // namespace mllm
