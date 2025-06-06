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

#include <memory>
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

static inline std::vector<std::shared_ptr<TensorValue>> wrapTensors2TensorIR(
    IRContext* ctx, const std::vector<Tensor>& tensors) {
  std::vector<std::shared_ptr<TensorValue>> tensor_ir_values;
  for (auto& t : tensors) {
    if (ctx->isCacheInputOutputTensor(t.uuid())) {
      tensor_ir_values.emplace_back(
          ctx->getCacheInputOutputTensor(t.uuid())->cast_<ir::tensor::TensorValue>());
    } else {
      auto ret = ctx->create<TensorValue>(t);
      ctx->cacheInputOutputTensor(t.uuid(), ret);
      tensor_ir_values.emplace_back(ret);
    }
  }
  return tensor_ir_values;
}
}  // namespace mllm::ir::tensor
