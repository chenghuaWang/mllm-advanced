/**
 * @file CopyOpEliminatePass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/IR/Tensor/Op.hpp"
#include "mllm/IR/Passes/CopyOpEliminatePass.hpp"
#include "mllm/Core/AOps/CopyOp.hpp"
#include "mllm/Utils/Common.hpp"
#include <algorithm>

namespace mllm::ir {

namespace MLLM_NAMESPACE_ANONYMOUS {
void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op);

void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx,
                   const ir::graph::SubGraphOp::self_ptr_t& subgraph_op) {
  auto region = subgraph_op->getTopRegion();
  auto rewriter = ir::IRWriter(ir_ctx, region);

  rewriter.walk<ir::linalg::CopyOp>(
      [&](ir::IRWriter& remover,
          const ir::linalg::CopyOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
        // Get mllm's op
        auto mllm_copy_op = (CopyOp*)op->getAOp();
        auto has_side_effect = mllm_copy_op->hasSideEffect();

        if (has_side_effect) {
          // NOTE: If has side effect, we need to maintain dst value, but can change the src value.
          auto inputs = op->inputs();
          auto dst_it = inputs.begin();
          auto src_it = std::next(inputs.begin());
          auto dst_value = (*dst_it)->cast_<ir::Val>();
          auto src_value = (*src_it)->cast_<ir::Val>();

          // Case 1:
          // auto C = A + B;
          // copy(D, C)
          // C will not be used again.
          if (src_value->consumerOps().size() == 1
              && src_value->consumerOps()[0]->isa_<ir::linalg::CopyOp>()) {
            // Replace the src->producer's output with dst value
            auto producer_op = src_value->producerOp();
            MLLM_RT_ASSERT(producer_op != nullptr);

            producer_op->replacePartialOutputs(src_value, dst_value);

            remover.removeValue(src_value);
            remover.removeOp(op);
          }
        } else {
          // TODO
          NYI("Impl a algorithm to detect if the copy op can be eliminated");
        }

        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });
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

uint8_t CopyOpEliminatePass::run(const ir::node_ptr_t& op) {
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

}  // namespace mllm::ir
