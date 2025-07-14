/**
 * @file Planning.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <vector>

#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm::nn::planning {

struct __Write2Context {
  inline __Write2Context write2(Tensor& alloced_tensor, int32_t pos = 0) {
    auto& TCB = MllmEngineCtx::instance().thisThread()->getTCB();
    TCB.planning_ctx_.registered_outs_.emplace_back(alloced_tensor, pos);
    return {};
  }

  inline void from(const Tensor& _) {
    // Release the TCB's planning context
    auto& TCB = MllmEngineCtx::instance().thisThread()->getTCB();
    TCB.planning_ctx_.registered_outs_.clear();
  }

  inline void from(const std::vector<Tensor>& _) {
    // Release the TCB's planning context
    auto& TCB = MllmEngineCtx::instance().thisThread()->getTCB();
    TCB.planning_ctx_.registered_outs_.clear();
  }
};

/**
 * @brief Usage:
 *
 * nn::planning::write2(C).from(A + B); // where C is pre-allocated tensor
 *
 * if op in from(...) is not an op that will alloc memory, this function will panic.
 * Those op include things like nn::split<k>, slice op.
 *
 * @note This function can only be used in eager mode.
 *
 * @param alloced_tensor
 * @param pos
 * @return __Write2Context
 */
inline __Write2Context write2(Tensor& alloced_tensor, int32_t pos = 0) {
  // `write2` can only be used in eager mode.
  // If in trace mode, it means you want to do static graph IR translation things.
  MLLM_RT_ASSERT_EQ(MllmEngineCtx::instance().traceMode(), false);

  auto& TCB = MllmEngineCtx::instance().thisThread()->getTCB();
  TCB.planning_ctx_.registered_outs_.emplace_back(alloced_tensor, pos);
  return {};
}

/**
 * @brief
 *
 * @note This function can be used in both eager and trace mode. But eager mode will not optimize
 * this. Trace mode will using some optimization pass to eliminate unnecessary memory allocation.
 *
 * @param dst
 * @param src
 * @param side_effect
 */
void copy(Tensor& dst, Tensor& src, bool side_effect = true);

}  // namespace mllm::nn::planning
