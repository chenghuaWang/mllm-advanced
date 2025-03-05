/**
 * @file Value.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/Tensor.hpp"
#include "mllm/IR/Builtin/Interface.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"
#include "mllm/IR/NodeRTTIClassOfImpl.hpp"

namespace mllm::ir::tensor {
class TensorIRValue : public Val {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorIRValue);

  ~TensorIRValue() override;
  TensorIRValue();
  explicit TensorIRValue(NodeKind kind);

  static inline bool classof(const Node* node) { RTTI_RK_VAL_TENSORIRVAL_IMPL(node); }
};

class TensorValue : public TensorIRValue, public SymbolInterface<TensorValue> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorValue);

  ~TensorValue() override;
  TensorValue();

  static self_ptr_t build(IRContext* ctx, const Tensor& tensor);

  static inline bool classof(const Node* node) { RTTI_RK_VAL_TENSORIRVAL_TENSORVAL_IMPL(node); }

  void dump(IRPrinter& p) override;

  Tensor tensor_;
};
}  // namespace mllm::ir::tensor
