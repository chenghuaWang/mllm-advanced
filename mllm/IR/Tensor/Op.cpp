/**
 * @file Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Tensor/Op.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"
#include "mllm/IR/Tensor/Value.hpp"

namespace mllm::ir::tensor {

TensorIROp::~TensorIROp() = default;

TensorIROp::TensorIROp() : Op(RK_Op_TensorIROp) {}

TensorIROp::TensorIROp(NodeKind kind) : Op(kind) {}

AllocOp::~AllocOp() = default;

AllocOp::AllocOp() : TensorIROp(RK_Op_TensorIROp_AllocOp) {}

AllocOp::self_ptr_t AllocOp::build(IRContext* ctx) {
  auto ret = std::make_shared<AllocOp>();

  // TODO

  return ret;
}

AllocOp::self_ptr_t AllocOp::build(IRContext* ctx, const val_ptr_t& val) {
  auto ret = std::make_shared<AllocOp>();
  (*ret)-- > val;
  return ret;
}

void AllocOp::dump(IRPrinter& p) {
  p.print("tensor.{}.AllocOp", deviceTypes2Str(getDevice()));
  Op::dump(p);
}

std::shared_ptr<ir::tensor::TensorValue> AllocOp::getAlloced() {
  return outputs().front()->cast_<ir::tensor::TensorValue>();
}

AllocGlobalOp::~AllocGlobalOp() = default;

AllocGlobalOp::AllocGlobalOp() : TensorIROp(RK_Op_TensorIROp_AllocGlobalOp) {}

AllocGlobalOp::self_ptr_t AllocGlobalOp::build(IRContext* ctx) {
  auto ret = std::make_shared<AllocGlobalOp>();
  return ret;
}

AllocGlobalOp::self_ptr_t AllocGlobalOp::build(IRContext* ctx, const val_ptr_t& val) {
  auto ret = std::make_shared<AllocGlobalOp>();

  (*ret)-- > val;

  return ret;
}

void AllocGlobalOp::dump(IRPrinter& p) {
  p.print("tensor.{}.AllocGlobalOp", deviceTypes2Str(getDevice()));
  Op::dump(p);
}

FreeOp::~FreeOp() = default;

FreeOp::FreeOp() : TensorIROp(RK_Op_TensorIROp_FreeOp) {}

FreeOp::self_ptr_t FreeOp::build(IRContext* ctx) {
  auto ret = std::make_shared<FreeOp>();

  // TODO

  return ret;
}

FreeOp::self_ptr_t FreeOp::build(IRContext* ctx, const val_ptr_t& val) {
  auto ret = std::make_shared<FreeOp>();
  (*val)-- > ret;
  return ret;
}

void FreeOp::dump(IRPrinter& p) {
  p.print("tensor.{}.FreeOp", deviceTypes2Str(getDevice()));
  Op::dump(p);
}

}  // namespace mllm::ir::tensor
