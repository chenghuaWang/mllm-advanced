/**
 * @file Thread.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <thread>
#include <utility>
#include <vector>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Engine/MemManager.hpp"
#include "mllm/Engine/SymbolTable.hpp"

namespace mllm {

// Planning context for MLLM Engine Thread Control Block (TCB)
// This context is for register pre-alloced buffer for op to use.
//
// e.g.:
// In eager mode: C = A + B will alloc a new memory buffer for C. But if you want to do D = A + B,
// which D is pre alloced by user, it becomes troubleshot. __MllmTCBPlanningCtx is involved to solve
// this problem. User can write code like this:
//
// nn::planning::write2(D).from(A + B);
struct __MllmTCBPlanningCtx {
  std::vector<std::pair<Tensor, int32_t>> registered_outs_;
};

struct MllmEngineTCB {
  MllmEngineTCB() : system_tid(std::this_thread::get_id()) {}

  __MllmTCBPlanningCtx planning_ctx_;
  std::thread::id system_tid;
};

class MllmEngineThread {
 public:
  [[nodiscard]] std::thread::id threadId() const;

  SymbolTable<std::string, std::shared_ptr<BaseOp>> layer_ops_table;

  inline MllmEngineTCB& getTCB() { return tcb_; }

 private:
  MllmEngineTCB tcb_;
};

}  // namespace mllm
