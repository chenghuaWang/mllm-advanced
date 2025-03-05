/**
 * @file Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Builtin/Attribute.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"

namespace mllm::ir::graph {

GraphIROp::~GraphIROp() = default;

GraphIROp::GraphIROp() : Op(RK_Op_GraphIROp) {}

GraphIROp::GraphIROp(const NodeKind& kind) : Op(kind) {}

SubGraphOp::~SubGraphOp() = default;

SubGraphOp::SubGraphOp() : GraphIROp(RK_Op_GraphIROp_SubGraphOp) {}

SubGraphOp::self_ptr_t SubGraphOp::build(IRContext* ctx,
                                         const std::shared_ptr<SymbolAttr>& symbol_attr,
                                         const std::shared_ptr<HierarchyBase>& hierarchy_base) {
  auto ret = std::make_shared<SubGraphOp>();
  ret->setSymbolAttr(symbol_attr);
  ret->createRegionAtTop();
  ret->hierarchy_base_ = hierarchy_base;

  ctx->addToSymbolTable(ret, symbol_attr->str());

  return ret;
}

void SubGraphOp::dump(IRPrinter& p) {
  p.print("graph.SubGraphOp @{} ", getSymbolAttr()->str());
  p.lbrace();

  getTopRegion()->dump(p);

  p.rbrace();
}

CallGraphOp::~CallGraphOp() = default;

CallGraphOp::CallGraphOp() : GraphIROp(RK_Op_GraphIROp_CallGraphOp) {}

void CallGraphOp::dump(IRPrinter& p) {
  p.print("graph.CallGraphOp @{} ", getSymbolAttr()->str());
  Op::dump(p);
}

CallGraphOp::self_ptr_t CallGraphOp::build(IRContext* ctx,
                                           const std::shared_ptr<SymbolAttr>& symbol_attr) {
  auto ret = std::make_shared<CallGraphOp>();
  ret->setSymbolAttr(symbol_attr);
  return ret;
}

}  // namespace mllm::ir::graph
