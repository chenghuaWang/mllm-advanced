/**
 * @file IR.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Engine/Context.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm::ir {

template<typename NnModule, typename... Args>
std::shared_ptr<IRContext> trace(NnModule& nn_module, Args&&... args) {
  MLLM_WARN("Mllm's IR does not support conditional operations, which means the trace function "
            "cannot be used to track dynamic control flow involving if statements, while "
            "loops, or other conditional constructs. This limitation is due to the static nature "
            "of Mllm's IR, which is designed to handle only static computational graphs "
            "without branching or dynamic behavior. Please ensure that your code does not contain "
            "any conditional logic when using the trace function. It is intended for static "
            "graphs where the execution path is fully determined at compile time. If your "
            "program relies on dynamic control flow, consider restructuring your code to avoid "
            "conditional statements or explore alternative tools that support dynamic tracing.");
  auto ir_ctx = std::make_shared<IRContext>();
  auto ir_module = ir_ctx->createAndSetModuleOp<ModuleOp>(ir_ctx->create<SymbolAttr>("main"));

  auto& ctx = MllmEngineCtx::instance();
  ctx.ir_context_ = ir_ctx;
  ctx.setTraceMode(true);
  nn_module.trace(std::forward<Args>(args)...);
  ctx.setTraceMode(false);
  return ir_ctx;
}

// TODO rewrite passes

}  // namespace mllm::ir
