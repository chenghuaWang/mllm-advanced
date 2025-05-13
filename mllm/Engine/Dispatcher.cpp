/**
 * @file Dispatcher.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <chrono>
#include <thread>
#include "mllm/Engine/Dispatcher.hpp"

namespace mllm {

void Dispatcher::setMllmEngineCtx(MllmEngineCtx* ctx) { engine_ctx_ = ctx; }

void Dispatcher::registerTaskSolver(const std::shared_ptr<ATaskSolver>& solver) {
  solvers_.reg(solver->task_type_, solver);
}

void Dispatcher::registerTaskFilter(const std::shared_ptr<TaskFilter>& task_filter) {
  task_filters_.push_back(task_filter);
}

bool Dispatcher::isConsumable(Task::ptr_t task) const {  // NOLINT
  for (auto& task_filter : task_filters_) {
    if (!task_filter->filt(engine_ctx_, task)) { return false; }
  }
  return true;
}

void Dispatcher::consume(Task::ptr_t task) { task->processed_ = true; }  // NOLINT

void Dispatcher::process(Task::ptr_t task) {  // NOLINT
  NYI("Dispatcher::process is not implemented yet");
}

void EagerDispatcher::process(Task::ptr_t task) {
  if (isConsumable(task)) {
    consume(task);
  } else {
    return;
  }
  if (!perf_) {
    task->op_->reshape(task->inputs_, task->outputs_);
    task->op_->setup(task->inputs_, task->outputs_);
    task->op_->forward(task->inputs_, task->outputs_);
  } else {
    auto start = std::chrono::high_resolution_clock::now();
    task->op_->reshape(task->inputs_, task->outputs_);
    task->op_->setup(task->inputs_, task->outputs_);
    task->op_->forward(task->inputs_, task->outputs_);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    task->exec_times_ = duration.count();
  }
}

void EagerDispatcher::setPerf(bool perf) { perf_ = perf; }

DispatcherManager::DispatcherManager(MllmEngineCtx* ctx) : engine_ctx_(ctx) {}

DispatcherManager::~DispatcherManager() {
  for (auto& [name, info] : dispatchers_._ref_raw_data()) {
    {
      std::lock_guard lock(info.thread_mutex_);
      info.is_running_ = false;
    }
    info.thread_notify_cv_.notify_all();
    if (info.attached_thread_.joinable()) { info.attached_thread_.join(); }
  }
}

void DispatcherManager::registerDispatcher(const std::string& name,
                                           const std::shared_ptr<Dispatcher>& dispatcher,
                                           uint8_t priority, bool separate_thread) {
  if (!separate_thread) {
    MLLM_WARN("separate_thread is set to false in registerDispatcher. This will block the "
              "DispatcherManager when dispatcher doing it work. Only use it when you are sure that "
              "there has only one dispatcher in dispatcher manager or there is no concurrent "
              "dispatching.");
    // unordered_map will auto create one meta info for us.
    MLLM_RT_ASSERT_EQ(dispatchers_.has(name), false);
    auto& info = dispatchers_._ref_raw_data()[name];
    info.dispatcher_impl_ = dispatcher;
    info.use_separate_thread_ = false;
    dispatcher->setMllmEngineCtx(engine_ctx_);
    priority_max_heap_.emplace(priority, name);
    return;
  }

  // unordered_map will auto create one meta info for us.
  MLLM_RT_ASSERT_EQ(dispatchers_.has(name), false);
  auto& info = dispatchers_._ref_raw_data()[name];
  info.dispatcher_impl_ = dispatcher;
  info.use_separate_thread_ = true;

  dispatcher->setMllmEngineCtx(engine_ctx_);

  priority_max_heap_.emplace(priority, name);
  info.attached_thread_ = std::thread([&info] {
    while (true) {
      std::unique_lock lock(info.thread_mutex_);
      info.thread_notify_cv_.wait(lock, [&] { return info.current_task_ || !info.is_running_; });

      if (!info.is_running_) break;

      if (info.current_task_) {
        info.dispatcher_impl_->process(info.current_task_);
        info.current_task_ = nullptr;
        info.current_promise_.set_value();
      }

      info.is_busy_ = false;
      lock.unlock();
      info.thread_notify_cv_.notify_all();
    }
  });
}

void DispatcherManager::finalizeAndFreeze() {
  // Freeze the dispatcher priority.
  while (!priority_max_heap_.empty()) {
    const auto& top = priority_max_heap_.top();
    priority_.emplace_back(top);
    priority_max_heap_.pop();
  }
}

void DispatcherManager::sendTask(Task::ptr_t task) {  // NOLINT
  for (const auto& detail_dispatcher_pair : priority_) {
    if (task->processed_) { break; }
    auto& info = dispatchers_[detail_dispatcher_pair.second];
    if (info.dispatcher_impl_->isConsumable(task)) { info.dispatcher_impl_->process(task); }
  }
}

std::future<void> DispatcherManager::sendAsyncTaskDirectTo(const std::string& dispatcher_name,
                                                           Task::ptr_t task) {  // NOLINT
  auto& info = dispatchers_[dispatcher_name];
  if (!info.dispatcher_impl_->isConsumable(task)) {  // NOLINT
    MLLM_WARN("Task in dispatcher {} is not consumable", dispatcher_name);
    std::promise<void> p;
    p.set_value();
    return p.get_future();
  }

  std::promise<void> task_promise;
  auto future = task_promise.get_future();

  std::unique_lock lock(info.thread_mutex_);
  info.thread_notify_cv_.wait(lock, [&] { return !info.is_busy_ || !info.is_running_; });

  if (!info.is_running_) {
    MLLM_ERROR_EXIT(kError, "Dispatcher is not running");
    return future;
  }

  info.is_busy_ = true;
  info.current_task_ = task;
  info.current_promise_ = std::move(task_promise);

  if (info.use_separate_thread_) {
    lock.unlock();
    info.thread_notify_cv_.notify_one();
  } else {
    lock.unlock();
    info.dispatcher_impl_->process(task);
    info.current_promise_.set_value();
    std::lock_guard finish_lock(info.thread_mutex_);
    info.is_busy_ = false;
  }
  return future;
}

}  // namespace mllm
