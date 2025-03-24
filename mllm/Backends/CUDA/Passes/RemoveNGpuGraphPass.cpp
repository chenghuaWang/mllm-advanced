/**
 * @file RemoveNGpuGraphPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/Passes/RemoveNGpuGraphPass.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::cuda {

uint8_t RemoveNonGpuGraphPass::run(const ir::node_ptr_t& op) {
  // the top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());
  r.walk<ir::graph::SubGraphOp>(
      [&](ir::IRWriter& remover,
          const std::shared_ptr<ir::graph::SubGraphOp>& op) -> ir::IRWriter::WalkResult {
        if (op->hierarchy_base_->device() != kCUDA) { remover.removeOp(op); }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::cuda
