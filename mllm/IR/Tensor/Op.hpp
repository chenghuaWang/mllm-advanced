/**
 * @file Op.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Node.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"
#include "mllm/IR/NodeRTTIClassOfImpl.hpp"
#include "mllm/IR/Tensor/Value.hpp"

namespace mllm::ir::tensor {

class TensorIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorIROp);

  ~TensorIROp() override;
  TensorIROp();
  explicit TensorIROp(NodeKind kind);

  /**
   * @brief Static inline function to check if the given Node pointer points to a TensorIROP type
   * object.
   *
   * This function uses RTTI (Run-Time Type Information) to implement the check for a TensorIROP
   * type object.
   *
   * @param node Pointer to the Node object to be checked.
   *
   * @return true if node points to a TensorIROP type object; false otherwise.
   */
  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_IMPL(node); }
};

class AllocOp : public TensorIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(AllocOp);

  ~AllocOp() override;
  AllocOp();

  static self_ptr_t build(IRContext* ctx);
  static self_ptr_t build(IRContext* ctx, const val_ptr_t& val);

  void dump(IRPrinter& p) override;

  std::shared_ptr<ir::tensor::TensorValue> getAlloced();

  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_ALLOCOP_IMPL(node); }
};

class AllocGlobalOp : public TensorIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(AllocGlobalOp);

  ~AllocGlobalOp() override;
  AllocGlobalOp();

  static self_ptr_t build(IRContext* ctx);
  static self_ptr_t build(IRContext* ctx, const val_ptr_t& val);

  void dump(IRPrinter& p) override;

  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_ALLOCGLOBALOP_IMPL(node); }
};

class FreeOp : public TensorIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(FreeOp);

  ~FreeOp() override;
  FreeOp();

  static self_ptr_t build(IRContext* ctx);
  static self_ptr_t build(IRContext* ctx, const val_ptr_t& val);

  void dump(IRPrinter& p) override;
  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_FREEOP_IMPL(node); }
};

}  // namespace mllm::ir::tensor