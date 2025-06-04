/**
 * @file TensorNamingPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief This pass will make tensor's name readable
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include "mllm/Backends/QNN/Passes/QnnTensorNamingPass.hpp"
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
  for (auto& _op : region->ops()) {
    // If has call graph op
    if (_op->isa_<ir::graph::CallGraphOp>()) {
      visitCallGraph(ir_ctx, _op->cast_<ir::graph::CallGraphOp>());
    } else if (_op->isa_<ir::linalg::LinalgIROp>()) {
      // The inputs is all named. We just need to name outputs.
      int cnt = 0;
      auto outputs = _op->outputs();

      for (auto& o : outputs) {
        MLLM_RT_ASSERT(o->isa_<ir::tensor::TensorValue>());
        auto t = o->cast_<ir::tensor::TensorValue>();

        if (t->hasSymbolAttr()) {
          t->name() += "." + t->getSymbolAttr()->str();
        } else {
          t->name() += "."
                       + std::string(opType2Str(_op->cast_<ir::linalg::LinalgIROp>()->getAOpType()))
                       + ".v." + std::to_string(cnt++);
        }
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

uint8_t QnnTensorNamingPass::run(const ir::node_ptr_t& op) {
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

  // Go through all ops start from call_main_graph_op
  // Give each call main op's inputs readable name
  {
    auto& inputs = call_main_graph_op->inputs();
    auto main_graph_name = call_main_graph_op->getSymbolAttr()->str();
    int cnt = 0;
    for (auto& i : inputs) {
      MLLM_RT_ASSERT(i->isa_<ir::tensor::TensorValue>());
      auto t = i->cast_<ir::tensor::TensorValue>();

      if (t->hasSymbolAttr()) {
        t->name() += "." + t->getSymbolAttr()->str();
      } else {
        t->name() += "." + main_graph_name + ".v." + std::to_string(cnt++);
      }
    }
  }

  // Visit all graph
  visitCallGraph(getCtx(), call_main_graph_op);

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::qnn
