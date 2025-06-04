/**
 * @file Value.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Core/DataTypes.hpp"
#include <string>

namespace mllm::ir::tensor {

TensorIRValue::~TensorIRValue() = default;

TensorIRValue::TensorIRValue() : Val(RK_Val_TensorIRVal) {}

TensorIRValue::TensorIRValue(NodeKind kind) : Val(kind) {}

TensorValue::~TensorValue() = default;

TensorValue::TensorValue() : TensorIRValue(RK_Val_TensorIRVal_TensorVal) {}

TensorValue::self_ptr_t TensorValue::build(IRContext* ctx, const Tensor& tensor) {
  auto ret = std::make_shared<TensorValue>();
  ret->tensor_ = tensor;
  ret->name() = std::to_string(tensor.uuid());

  // If this tensor is parameter tensor or global tensor which has name
  switch (tensor.memType()) {
    case kGlobal:
    case kParams: ret->setSymbolAttr(SymbolAttr::build(ctx, tensor.name())); break;
    default: break;
  }

  return ret;
}

void TensorValue::dump(IRPrinter& p) {
  Val::dump(p);

  p.print("tensor");
  IRPrinter::langle();

  // shape
  {
    IRPrinter::lsbracket();
    auto size = tensor_.shape().size();
    for (int i = 0; i < size; ++i) {
      p.print("{}", tensor_.shape()[i]);
      if (i < size - 1) IRPrinter::comma();
    }
    IRPrinter::rsbracket();
  }

  IRPrinter::comma();

  // dtype
  p.print("{}", dataTypes2Str(tensor_.dtype()));

  IRPrinter::comma();

  // device type
  p.print("{}", deviceTypes2Str(tensor_.device()));

  IRPrinter::rangle();

  if (hasSymbolAttr()) {
    IRPrinter::lsbracket();
    p.print("@{}", getSymbolAttr()->str());
    IRPrinter::rsbracket();
  }
}

}  // namespace mllm::ir::tensor
