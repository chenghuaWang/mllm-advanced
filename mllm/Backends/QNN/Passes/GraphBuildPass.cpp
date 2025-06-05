/**
 * @file GraphBuildPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Passes/GraphBuildPass.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/Utils/Common.hpp"

#include <algorithm>

namespace mllm::qnn {

uint8_t GraphBuildPass::run(const ir::node_ptr_t& op) {
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
          graphs_ir_to_be_compiled.push_back(op);
        }

        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  MLLM_RT_ASSERT_EQ(graph_need_to_be_compiled_.size(), graphs_ir_to_be_compiled.size())

  // Verify all subgraph has no CallGraphOp
  for (auto& graph_ir : graphs_ir_to_be_compiled) {
    auto graph_ir_r = ir::IRWriter(getCtx(), graph_ir->getTopRegion());
    r.walk<ir::graph::CallGraphOp>(
        [&](ir::IRWriter& reader,
            const ir::graph::CallGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
          MLLM_ERROR_EXIT(kError, "You should call GraphInlinePass before GraphBuildPass. And for "
                                  "now, manually inline CallGraphOp is highly recommended.");

          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });
  }

  // QNN Graph build and Compile
  for (auto& graph_ir : graphs_ir_to_be_compiled) {}
  // TODO

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::qnn