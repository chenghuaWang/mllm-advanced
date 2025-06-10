/**
 * @file QnnMarkIOTensorPass.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-06-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Passes/QnnMarkIOTensorPass.hpp"
#include "mllm/IR/Builtin/Attribute.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/IR/CF/Op.hpp"

namespace mllm::qnn {

namespace MLLM_NAMESPACE_ANONYMOUS {
void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op);
void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                   const ir::graph::SubGraphOp::self_ptr_t& subgraph_op) {
  for (auto& i : subgraph_op->inputs()) {
    i->setAttr("is_graph_input", ir_ctx->create<ir::BoolAttr>(true));
  }
  auto region = subgraph_op->getTopRegion();
  for (auto& _op : region->ops()) {
    // If has call graph op
    if (_op->isa_<ir::graph::CallGraphOp>()) {
      visitCallGraph(ir_ctx, _op->cast_<ir::graph::CallGraphOp>());
    } else if (_op->isa_<ir::cf::ReturnOp>()) {
      // The inputs value of this cf::ReturnOp is the output of the subgraph
      for (auto& i : _op->inputs()) {
        i->setAttr("is_graph_output", ir_ctx->create<ir::BoolAttr>(true));
      }
    }
  }
}

void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op) {
  // Panic if input of call graph has no name
  auto& inputs = call_op->inputs();

  visitSubGraph(
      ir_ctx,
      ir_ctx->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>());
}
}  // namespace MLLM_NAMESPACE_ANONYMOUS

uint8_t QnnMarkIOTensorPass::run(const ir::node_ptr_t& op) {
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
