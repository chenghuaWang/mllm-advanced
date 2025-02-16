/**
 * @file Pass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::ir {

uint8_t Pass::run(const node_ptr_t& op) { return PASS_RET_SUCCESS; }

void Pass::setCtx(const std::shared_ptr<IRContext>& ctx) { ctx_ = ctx; }

std::shared_ptr<IRContext> Pass::getCtx() { return ctx_; }

}  // namespace mllm::ir