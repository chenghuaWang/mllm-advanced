/**
 * @file ElewiseOps.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/ElewiseOp.hpp"

namespace mllm::arm {

class ArmAddOp : public AddOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmSubOp : public AddOp {};

class ArmMulOp : public AddOp {};

class ArmDivOp : public AddOp {};

class ArmAddOpFactory final : public TypedOpFactory<OpType::kAdd, AddOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const AddOpCargo& cargo) override {
    return std::make_shared<ArmAddOp>();
  }
};

class ArmSubOpFactory : public TypedOpFactory<OpType::kSub, SubOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SubOpCargo& cargo) override {
    // TODO
    return nullptr;
  }
};

class ArmMulOpFactory : public TypedOpFactory<OpType::kMul, MulOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const MulOpCargo& cargo) override {
    // TODO
    return nullptr;
  }
};

class ArmDivOpFactory : public TypedOpFactory<OpType::kDiv, DivOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const DivOpCargo& cargo) override {
    // TODO
    return nullptr;
  }
};

}  // namespace mllm::arm
