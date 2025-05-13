/**
 * @file Dispatcher.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <queue>
#include <cstdint>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Engine/SymbolTable.hpp"

namespace mllm {

class MllmEngineCtx;

enum TaskType : uint8_t {
  kTaskType_Start = 0,
  kTaskType_None,
  kDispatchOpTask,
  kBeginGraphTask,
  kEndGraphTask,
  kTaskType_End,
};

struct Task {
  using ptr_t = std::shared_ptr<Task>;

  double exec_times_ = -1;
  std::shared_ptr<BaseOp> op_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  TaskType task_type_ = kTaskType_None;
  bool processed_ = false;
};

class TaskFilter {
 public:
  virtual bool filt(MllmEngineCtx* ctx, Task::ptr_t task) = 0;
};

class ATaskSolver {
 public:
  virtual void process(MllmEngineCtx* ctx, Task::ptr_t task) = 0;

  TaskType task_type_ = kTaskType_None;
};

class Dispatcher {
 public:
  void setMllmEngineCtx(MllmEngineCtx* ctx);

  void registerTaskSolver(const std::shared_ptr<ATaskSolver>& solver);

  void registerTaskFilter(const std::shared_ptr<TaskFilter>& task_filter);

  [[nodiscard]] bool isConsumable(Task::ptr_t task) const;

  void consume(Task::ptr_t task);

  virtual void process(Task::ptr_t task);

 private:
  SymbolTable<TaskType, std::shared_ptr<ATaskSolver>> solvers_;
  std::vector<std::shared_ptr<TaskFilter>> task_filters_;
  MllmEngineCtx* engine_ctx_ = nullptr;
};

class EagerDispatcher final : public Dispatcher {
 public:
  void process(Task::ptr_t task) override;

  void setPerf(bool perf);

 private:
  bool perf_ = false;
};

class TraceDispatcher final : public Dispatcher {
 public:
  void process(Task::ptr_t task) override;
};

// Group Dispatcher manage a group of same dispatchers. It will send task to available dispatcher in
// the dispatcher group
class GroupDispatcherDispatcher final : public Dispatcher {
 public:
  void process(Task::ptr_t task) override;
};

class DispatcherManager {
  struct DispatcherMetaInfo {
    std::promise<void> current_promise_;
    Task::ptr_t current_task_ = nullptr;
    std::shared_ptr<Dispatcher> dispatcher_impl_ = nullptr;
    std::thread attached_thread_;
    std::mutex thread_mutex_;
    std::condition_variable thread_notify_cv_;
    bool is_busy_ = false;
    bool use_separate_thread_ = false;
    bool is_running_ = false;
    uint8_t priority_ = 0;
  };

 public:
  explicit DispatcherManager(MllmEngineCtx* ctx);

  ~DispatcherManager();

  void registerDispatcher(const std::string& name, const std::shared_ptr<Dispatcher>& dispatcher,
                          uint8_t priority, bool separate_thread = false);

  void finalizeAndFreeze();

  void sendTask(Task::ptr_t task);

  std::future<void> sendAsyncTaskDirectTo(const std::string& dispatcher_name, Task::ptr_t task);

 private:
  struct ComparePair {
    bool operator()(const std::pair<uint8_t, std::string>& a,
                    const std::pair<uint8_t, std::string>& b) {
      return a.first < b.first;
    }
  };

  std::priority_queue<std::pair<uint8_t, std::string>, std::vector<std::pair<uint8_t, std::string>>,
                      ComparePair>
      priority_max_heap_;  // The bigger the priority, the earlier the dispatcher will be executed.
  std::vector<std::pair<uint8_t, std::string>> priority_;
  SymbolTable<std::string, DispatcherMetaInfo> dispatchers_;
  MllmEngineCtx* engine_ctx_ = nullptr;
};

}  // namespace mllm
