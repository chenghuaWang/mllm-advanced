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
#include "mllm/Backends/QNN/Passes/QnnAnonymousOpNamingPass.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::qnn {

uint8_t QnnAnonymousOpNamingPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::qnn
