/**
 * @file Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/CF/Op.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"

namespace mllm::ir::cf {
ControlFlowIROp::ControlFlowIROp() : Op(RK_Op_ControlFlowIROp) {}

ControlFlowIROp::ControlFlowIROp(NodeKind kind) : Op(kind) {}

ControlFlowIROp::~ControlFlowIROp() = default;

ReturnOp::ReturnOp() : ControlFlowIROp(RK_Op_ControlFlowIROp_ReturnOp) {}

ReturnOp::~ReturnOp() = default;

ReturnOp::self_ptr_t ReturnOp::build(IRContext* ctx, const val_ptr_t& val) {
  auto ret = std::make_shared<ReturnOp>();
  (*val)-- > ret;
  return ret;
}

ReturnOp::self_ptr_t ReturnOp::build(IRContext* ctx, const std::vector<val_ptr_t>& vals) {
  auto ret = std::make_shared<ReturnOp>();
  for (auto& val : vals) (*val)-- > ret;
  return ret;
}

void ReturnOp::dump(IRPrinter& p) {
  p.print("cf.ReturnOp");
  Op::dump(p);
}
}  // namespace mllm::ir::cf
