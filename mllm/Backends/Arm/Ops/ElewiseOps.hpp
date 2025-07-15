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

class ArmAddOp final : public AddOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmSubOp final : public SubOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmMulOp final : public MulOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmDivOp final : public DivOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmNegOp final : public DivOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmAddOpFactory final : public TypedOpFactory<OpType::kAdd, AddOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const AddOpCargo& cargo) override {
    return std::make_shared<ArmAddOp>();
  }
};

class ArmSubOpFactory : public TypedOpFactory<OpType::kSub, SubOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SubOpCargo& cargo) override {
    return std::make_shared<ArmSubOp>();
  }
};

class ArmMulOpFactory : public TypedOpFactory<OpType::kMul, MulOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const MulOpCargo& cargo) override {
    return std::make_shared<ArmMulOp>();
  }
};

class ArmDivOpFactory : public TypedOpFactory<OpType::kDiv, DivOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const DivOpCargo& cargo) override {
    return std::make_shared<ArmDivOp>();
  }
};

class ArmNegOpFactory : public TypedOpFactory<OpType::kNeg, NegOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const NegOpCargo& cargo) override {
    return std::make_shared<ArmNegOp>();
  }
};

}  // namespace mllm::arm
