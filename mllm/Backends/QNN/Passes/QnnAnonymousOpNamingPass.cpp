/**
 * @file QnnAnonymousOpNamingPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include <unordered_map>
#include "mllm/Backends/QNN/Passes/QnnAnonymousOpNamingPass.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::qnn {

namespace MLLM_NAMESPACE_ANONYMOUS {
void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op);

void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                   const ir::graph::SubGraphOp::self_ptr_t& subgraph_op) {
  auto region = subgraph_op->getTopRegion();

  std::unordered_map<OpType, int32_t> intra_optype_cnt;

  for (auto& _op : region->ops()) {
    // If has call graph op
    if (_op->isa_<ir::graph::CallGraphOp>()) {
      visitCallGraph(ir_ctx, _op->cast_<ir::graph::CallGraphOp>());
    } else if (_op->isa_<ir::linalg::LinalgIROp>()) {
      auto mllm_op = _op->cast_<ir::linalg::LinalgIROp>()->getAOp();
      auto mllm_op_type = _op->cast_<ir::linalg::LinalgIROp>()->getAOpType();

      // If this op has no name, it means that this op is called from function not nn.layer.
      if (mllm_op->name().empty()) {
        if (!intra_optype_cnt.count(mllm_op_type)) { intra_optype_cnt.insert({mllm_op_type, -1}); }
        intra_optype_cnt[mllm_op_type] += 1;

        mllm_op->setName(subgraph_op->getSymbolAttr()->str() + "." + opType2Str(mllm_op_type) + "."
                         + std::to_string(intra_optype_cnt[mllm_op_type]));
      }
    } else {
      // TODO Error
    }
  }
}

void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op) {
  // Panic if input of call graph has no name
  auto& inputs = call_op->inputs();
  for (auto& input : inputs) {
    MLLM_RT_ASSERT(input->isa_<ir::tensor::TensorValue>()
                   && !input->cast_<ir::tensor::TensorValue>()->name().empty());
  }

  visitSubGraph(
      ir_ctx,
      ir_ctx->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>());
}
}  // namespace MLLM_NAMESPACE_ANONYMOUS

uint8_t QnnAnonymousOpNamingPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  // Find the top CallGraphOp
  ir::graph::CallGraphOp::self_ptr_t call_main_graph_op = nullptr;
  r.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& remover,
          const ir::graph::CallGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
        // Make sure there is only one call graph op in the ModuleOp
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);

        call_main_graph_op = op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Visit all graph
  visitCallGraph(getCtx(), call_main_graph_op);

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::qnn
