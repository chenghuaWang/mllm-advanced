/**
 * @file KaiQuantizationPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Passes/KaiQuantizationPass.hpp"
#include "mllm/IR/Passes/Pass.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

uint8_t KaiQuantizationPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // TODO

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::arm
