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

#include <memory>
#include "mllm/IR/Builtin/Interface.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Builtin/Attribute.hpp"
#include "mllm/Nn/HierarchyBase.hpp"

namespace mllm::ir::graph {
class GraphIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(GraphIROp);

  ~GraphIROp() override;
  GraphIROp();
  explicit GraphIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_GRAPHIROP_IMPL(node); }
};

class SubGraphOp : public GraphIROp, public SymbolInterface<SubGraphOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(SubGraphOp);

  ~SubGraphOp() override;
  SubGraphOp();

  static self_ptr_t build(IRContext* ctx, const std::shared_ptr<SymbolAttr>& symbol_attr,
                          const std::shared_ptr<HierarchyBase>& hierarchy_base);

  void dump(IRPrinter& p) override;

  std::shared_ptr<HierarchyBase> hierarchy_base_ = nullptr;

  static inline bool classof(const Node* node) { RTTI_RK_OP_GRAPHIROP_SUBGRAPHOP_IMPL(node); }
};

class CallGraphOp : public GraphIROp, public SymbolInterface<CallGraphOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(CallGraphOp);

  ~CallGraphOp() override;
  CallGraphOp();

  void dump(IRPrinter& p) override;

  static self_ptr_t build(IRContext* ctx, const std::shared_ptr<SymbolAttr>& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_GRAPHIROP_CALLGRAPHOP_IMPL(node); }
};
}  // namespace mllm::ir::graph
