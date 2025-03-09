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

uint8_t RemoveNonGpuGraphPass::run(const node_ptr_t& op) {
  // the top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ModuleOp>());

  auto r = IRWriter(getCtx(), op->cast_<ModuleOp>()->getTopRegion());
  r.walk<graph::SubGraphOp>(
      [&](IRWriter& remover, const std::shared_ptr<graph::SubGraphOp>& op) -> IRWriter::WalkResult {
        if (op->hierarchy_base_->device() != kCUDA) { remover.removeOp(op); }
        return IRWriter::WalkResult::WALK_CONTINUE;
      });

  return PASS_RET_SUCCESS;
}

}  // namespace mllm::cuda
