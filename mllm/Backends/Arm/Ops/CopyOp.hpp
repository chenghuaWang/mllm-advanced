/**
 * @file CopyOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/CopyOp.hpp"

namespace mllm::arm {

class ArmCopyOp final : public CopyOp {
 public:
  explicit ArmCopyOp(const CopyOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmCopyOpFactory : public TypedOpFactory<OpType::kCopy, CopyOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const CopyOpCargo& cargo) override {
    return std::make_shared<ArmCopyOp>(cargo);
  }
};

}  // namespace mllm::arm