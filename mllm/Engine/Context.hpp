/**
 * @file Context.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Engine/BackendBase.hpp"
#include "mllm/Engine/MemManager.hpp"
#include "mllm/Engine/SymbolTable.hpp"
#include "mllm/Engine/Thread.hpp"
#include "mllm/Engine/Dispatcher.hpp"

namespace mllm {

class MllmEngineCtx {
 public:
  static MllmEngineCtx& instance() {
    static MllmEngineCtx instance;
    return instance;
  }

  MllmEngineCtx();

  [[nodiscard]] bool traceMode() const;

  void setTraceMode(bool trace_mode);

  std::shared_ptr<MllmEngineThread> thisThread();

  std::shared_ptr<MllmEngineThread> mainThread();

  MllmEngineCtx(const MllmEngineCtx&) = delete;

  MllmEngineCtx& operator=(const MllmEngineCtx&) = delete;

  uint32_t getUUID();

  std::shared_ptr<MemManager> mem() const;

  void registerBackend(const std::shared_ptr<BackendBase>& new_backend);

  std::shared_ptr<BackendBase> getBackend(DeviceTypes device);

  /// Directly dispatch op forward require.
  /// Will not go through dispatcher pipeline.
  std::vector<Tensor> dispatch(const std::string& name, const std::vector<Tensor>& inputs);

  /// Directly dispatch op forward require.
  /// Will not go through dispatcher pipeline.
  std::vector<Tensor> dispatch(OpType op_type, const BaseOpCargoBase& base_cargo,
                               const std::vector<Tensor>& inputs);

  std::vector<Tensor> sendTask2Dispatcher(const std::string& name,
                                          const std::vector<Tensor>& inputs);

  std::vector<Tensor> sendTask2Dispatcher(OpType op_type, const BaseOpCargoBase& base_cargo,
                                          const std::vector<Tensor>& inputs);

  std::vector<Tensor> sendAsyncTask2DispatcherAndWait(const std::string& dispatcher_name,
                                                      const std::string& name,
                                                      const std::vector<Tensor>& inputs);

  std::vector<Tensor> sendAsyncTask2DispatcherAndWait(const std::string& dispatcher_name,
                                                      OpType op_type,
                                                      const BaseOpCargoBase& base_cargo,
                                                      const std::vector<Tensor>& inputs);

  inline DispatcherManager& getDispatcherManager() { return dispatcher_manager_; }

  void shutdown();

  bool perf_ = false;

  // param info place
  std::shared_ptr<void> ir_context_ = nullptr;

 private:
  // backend
  SymbolTable<DeviceTypes, std::shared_ptr<BackendBase>> backends_table_;

  // uuid
  std::atomic<uint32_t> custom_uuid_giver_ = 0;

  bool trace_mode_ = false;  // default in eager mode.

  std::shared_ptr<MllmEngineThread> main_thread_;
  std::shared_ptr<MemManager> main_thread_mem_;

  // NOTE: only main thread can creates other threads. Which means that thread_map_ has no meanings
  // to be thread safe.
  std::unordered_map<std::thread::id, std::shared_ptr<MllmEngineThread>> thread_map_;

  // Dispatcher Manager
  DispatcherManager dispatcher_manager_;
};

}  // namespace mllm
