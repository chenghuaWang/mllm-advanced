/**
 * @file QnnGraphBuildPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Passes/QnnGraphBuildPass.hpp"
#include "mllm/Backends/QNN/Ops/ElewiseOp.hpp"
#include "mllm/Backends/QNN/Ops/LinearOp.hpp"
#include "mllm/Backends/QNN/Ops/MatMulOp.hpp"
#include "mllm/Backends/QNN/Ops/SiLUOp.hpp"
#include "mllm/Backends/QNN/QnnBackend.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Utils/Common.hpp"

#include <algorithm>
#include <memory>

namespace mllm::qnn {

QnnGraphBuildPass::QnnGraphBuildPass() {
  regPattern<QnnMatMulOpPattern, QnnLinearOpPattern, QnnSiLUOpPattern, QnnAddOpPattern,
             QnnSubOpPattern, QnnMulOpPattern, QnnDivOpPattern>();
}

uint8_t QnnGraphBuildPass::run(const ir::node_ptr_t& op) {
  // Give pattern the context pointer
  for (auto& p : patterns_) { p.second->setIRCtx(getCtx()); }

  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());
  std::vector<ir::graph::SubGraphOp::self_ptr_t> graphs_ir_to_be_compiled;
  r.walk<ir::graph::SubGraphOp>(
      [&](ir::IRWriter& reader,
          const ir::graph::SubGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
        auto sub_graph_name = op->getSymbolAttr()->str();
        if (std::ranges::find(graph_need_to_be_compiled_, sub_graph_name)
            != graph_need_to_be_compiled_.end()) {
          MLLM_RT_ASSERT_EQ(op->getDevice(), DeviceTypes::kQNN);
          graphs_ir_to_be_compiled.push_back(op);
        }

        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  MLLM_RT_ASSERT_EQ(graph_need_to_be_compiled_.size(), graphs_ir_to_be_compiled.size())

  // Verify all subgraph has no CallGraphOp
  for (auto& graph_ir : graphs_ir_to_be_compiled) {
    auto graph_ir_r = ir::IRWriter(getCtx(), graph_ir->getTopRegion());
    graph_ir_r.walk<ir::graph::CallGraphOp>(
        [&](ir::IRWriter& reader,
            const ir::graph::CallGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
          MLLM_ERROR_EXIT(kError, "You should call GraphInlinePass before GraphBuildPass. And for "
                                  "now, manually inline CallGraphOp is highly recommended.");

          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });
  }

  // QNN Graph build and Compile
  for (auto& graph_ir : graphs_ir_to_be_compiled) { buildQnnLego(graph_ir); }

  return ir::PASS_RET_SUCCESS;
}

void QnnGraphBuildPass::buildQnnLego(const ir::graph::SubGraphOp::self_ptr_t& sub_graph_op) {
  auto& mllm_ctx = mllm::MllmEngineCtx::instance();
  auto qnn_backend = std::static_pointer_cast<QnnBackend>(mllm_ctx.getBackend(kQNN));
  auto g = qnn_backend->createQnnGraph(sub_graph_op->getSymbolAttr()->str(), sub_graph_op,
                                       qnn_backend->htpFuncSymbols(), qnn_backend->htpBackend());

  std::vector<ir::tensor::TensorValue::self_ptr_t> g_inputs;
  std::vector<ir::tensor::TensorValue::self_ptr_t> g_outputs;

  for (auto& i : sub_graph_op->getTopRegion()->inputs()) {
    MLLM_RT_ASSERT_EQ(i->isa_<ir::tensor::TensorValue>(), true);
    g_inputs.emplace_back(i->cast_<ir::tensor::TensorValue>());
  }

  for (auto& o : sub_graph_op->getTopRegion()->outputs()) {
    MLLM_RT_ASSERT_EQ(o->isa_<ir::tensor::TensorValue>(), true);
    g_outputs.emplace_back(o->cast_<ir::tensor::TensorValue>());
  }

  g->startRecord();
  g->setupInputs(g_inputs);

  auto graph_region = sub_graph_op->getTopRegion();
  for (auto& op : graph_region->ops()) {
    if (op->isa_<ir::graph::CallGraphOp>()) {
      MLLM_ERROR_EXIT(kError,
                      "When building QNN graph, found CallGraphOp in Subgraph. You should call "
                      "QnnGraphInlinePass first to inline all nested qnn graph into one graph.");
    } else if (op->isa_<ir::linalg::LinalgIROp>()) {
      auto mllm_op = op->cast_<ir::linalg::LinalgIROp>()->getAOp();
      auto mllm_op_type = op->cast_<ir::linalg::LinalgIROp>()->getAOpType();

      MLLM_RT_ASSERT_EQ(patterns_.count(mllm_op_type), 1);
      MLLM_RT_ASSERT_EQ(patterns_[mllm_op_type]->match(op), true);

      std::vector<ir::tensor::TensorValue::self_ptr_t> op_inputs;
      std::vector<ir::tensor::TensorValue::self_ptr_t> op_outputs;

      for (auto& i : op->inputs()) {
        MLLM_RT_ASSERT_EQ(i->isa_<ir::tensor::TensorValue>(), true);
        op_inputs.emplace_back(i->cast_<ir::tensor::TensorValue>());
      }

      for (auto& o : op->outputs()) {
        MLLM_RT_ASSERT_EQ(o->isa_<ir::tensor::TensorValue>(), true);
        op_outputs.emplace_back(o->cast_<ir::tensor::TensorValue>());
      }

      MLLM_RT_ASSERT_EQ(patterns_[mllm_op_type]->addNode(*g, op, op_inputs, op_outputs), true);
    } else {
      NYI("This op is not supported by mllm qnn backend yet or this op has no need to be compiled");
    }
  }

  g->setupOutputs(g_outputs);  // TODO
  g->endRecord();
  g->compile();
}

}  // namespace mllm::qnn