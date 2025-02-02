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

namespace mllm::X86 {

class X86AddOp : public AddOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class X86SubOp : public AddOp {};

class X86MulOp : public AddOp {};

class X86DivOp : public AddOp {};

class X86AddOpFactory final : public TypedOpFactory<OpType::kAdd, AddOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const AddOpCargo& cargo) override {
    return std::make_shared<X86AddOp>();
  }
};

class X86SubOpFactory : public TypedOpFactory<OpType::kSub, SubOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SubOpCargo& cargo) override {
    // TODO
    return nullptr;
  }
};

class X86MulOpFactory : public TypedOpFactory<OpType::kMul, MulOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const MulOpCargo& cargo) override {
    // TODO
    return nullptr;
  }
};

class X86DivOpFactory : public TypedOpFactory<OpType::kDiv, DivOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const DivOpCargo& cargo) override {
    // TODO
    return nullptr;
  }
};

}  // namespace mllm::X86
