/**
 * @file QnnGraphInlinePass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include "mllm/Backends/QNN/Passes/QnnGraphInlinePass.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/CF/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::qnn {

namespace MLLM_NAMESPACE_ANONYMOUS {

void inlineSubgraph(const std::shared_ptr<ir::IRContext>& ctx,
                    const ir::graph::SubGraphOp::self_ptr_t& op) {
  auto region = op->getTopRegion();
  auto r = ir::IRWriter(ctx, region);

  r.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& rewriter,
          const ir::graph::CallGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
        // Recursively inline subgraph.
        auto callee_g =
            ctx->lookupSymbolTable(op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
        inlineSubgraph(ctx, callee_g);
        auto callee_region = callee_g->getTopRegion();

        for (auto& callee_region_op : callee_region->ops()) {
          if (!callee_region_op->isa_<ir::cf::ReturnOp>()) {
            rewriter.insertOpAtPos(op, ir::IRWriter::BEFORE, callee_region_op);
          }
        }

        rewriter.removeOp(op);
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });
}

}  // namespace MLLM_NAMESPACE_ANONYMOUS

uint8_t QnnGraphInlinePass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  if (graph_need_to_be_inlined_.size() == 0) {
    // TODO inline all qnn graph greedy.
  } else {
    std::vector<ir::graph::SubGraphOp::self_ptr_t> qnn_graphs_wait_for_inlining;
    for (auto& name : graph_need_to_be_inlined_) {
      auto g = getCtx()->lookupSymbolTable(name);
      MLLM_RT_ASSERT(g != nullptr);
      MLLM_RT_ASSERT(g->isa_<ir::graph::SubGraphOp>());
      MLLM_RT_ASSERT(g->cast_<ir::graph::SubGraphOp>()->getDevice() == kQNN);
      qnn_graphs_wait_for_inlining.push_back(g->cast_<ir::graph::SubGraphOp>());
    }
    for (auto& g : qnn_graphs_wait_for_inlining) { inlineSubgraph(getCtx(), g); }
  }

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::qnn
