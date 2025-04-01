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

#include <cstdint>
#include <vector>
#include <memory>
#include "mllm/Engine/SymbolTable.hpp"

namespace mllm {

class MllmEngineCtx;

enum TaskType : uint8_t {
  kTaskType_Start = 0,
  kTaskType_None,
  kDispatchOpTask,
  kEnterModuleTask,
  kExitModuleTask,
  kDispatchModuleTask,
  kTaskType_End,
};

class Task {
 private:
  TaskType type_ = kTaskType_None;
  bool processed_ = false;
};

class TaskFilter {
 public:
  virtual bool filt(const Task& task) = 0;
};

class ATaskSolver {
 public:
  virtual void process(const Task& task) = 0;

 private:
};

class EagerOpDispatchTaskSolver final : public ATaskSolver {
 public:
  void process(const Task& task) override;

 private:
};

std::shared_ptr<EagerOpDispatchTaskSolver> createEagerOpDispatchTaskSolver();

class TraceOpDispatchTaskSolver final : public ATaskSolver {
 public:
  void process(const Task& task) override;

 private:
};

std::shared_ptr<TraceOpDispatchTaskSolver> createTraceOpDispatchTaskSolver();

class Dispatcher {
 public:
  void registerTaskSolver(const std::shared_ptr<ATaskSolver>& solver);

  void registerTaskFilter(const std::shared_ptr<TaskFilter>& task_filter);

  virtual void process(const Task& task);

 private:
  SymbolTable<TaskType, std::shared_ptr<ATaskSolver>> solvers_;
  std::vector<std::shared_ptr<TaskFilter>> task_filters_;
  MllmEngineCtx* engine_ctx_ = nullptr;
};

class EagerDispatcher final : public Dispatcher {};

class TraceDispatcher final : public Dispatcher {};

class DispatcherManager {};

}  // namespace mllm
