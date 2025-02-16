/**
 * @file Op.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/IR/Builtin/Interface.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"
#include "mllm/IR/Builtin/Attribute.hpp"
#include "mllm/IR/NodeRTTIClassOfImpl.hpp"
#include "mllm/Utils/IRPrinter.hpp"

namespace mllm::ir {

class BuiltinIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(BuiltinIROp);

  ~BuiltinIROp() override;
  BuiltinIROp();
  explicit BuiltinIROp(NodeKind kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_BUILTINIROP_IMPL(node); }
};

class ModuleOp : public BuiltinIROp, public SymbolInterface<ModuleOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ModuleOp);

  ~ModuleOp() override;
  ModuleOp();

  void dump(IRPrinter& p) override final;

  static self_ptr_t build(IRContext* ctx, const std::shared_ptr<SymbolAttr>& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_BUILTINIROP_MODULEOP_IMPL(node); }
};

}  // namespace mllm::ir