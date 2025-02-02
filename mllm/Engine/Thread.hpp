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
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Engine/MemManager.hpp"
#include "mllm/Engine/SymbolTable.hpp"

namespace mllm {

struct MllmEngineTCB {
  MllmEngineTCB() : system_tid(std::this_thread::get_id()) {}

  std::thread::id system_tid;
};

class MllmEngineThread {
 public:
  [[nodiscard]] std::thread::id threadId() const;

  SymbolTable<std::string, std::shared_ptr<BaseOp>> layer_ops_table;

 private:
  MllmEngineTCB tcb_;
};

}  // namespace mllm
