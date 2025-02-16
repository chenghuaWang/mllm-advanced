/**
 * @file Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"

namespace mllm::ir {

BuiltinIROp::~BuiltinIROp() = default;

BuiltinIROp::BuiltinIROp() : Op(RK_Op_BuiltinIROp) {}

BuiltinIROp::BuiltinIROp(NodeKind kind) : Op(kind) {}

ModuleOp::~ModuleOp() = default;

ModuleOp::ModuleOp() : BuiltinIROp(RK_Op_BuiltinIROp_ModuleOp) {}

ModuleOp::self_ptr_t ModuleOp::build(IRContext* ctx,
                                     const std::shared_ptr<SymbolAttr>& symbol_attr) {
  auto ret = std::make_shared<ModuleOp>();

  ret->setSymbolAttr(symbol_attr);
  ret->createRegionAtTop();

  return ret;
}

void ModuleOp::dump(IRPrinter& p) {
  p.print("@{} ", getSymbolAttr()->str());
  getTopRegion()->dump(p);
  p.newline();
}

}  // namespace mllm::ir