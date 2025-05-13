/**
 * @file ThreadPool.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <pthread.h>
#include "mllm/Utils/ThreadPool.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

thread_local int MllmThreadPool::current_thread_id_ = -1;

MllmThreadPool::~MllmThreadPool() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
}

void MllmThreadPool::initialize(int num_threads) {
  stop_ = false;
  num_threads_ = num_threads;
  if (num_threads_ > std::thread::hardware_concurrency()) {
    MLLM_ERROR_EXIT(kError,
                    "This device only supports {} threads, but you want to create {} threads.",
                    std::thread::hardware_concurrency(), num_threads_);
  }

  workers_.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    workers_.emplace_back([this, i] {
      current_thread_id_ = i;
      workLoop();
    });
  }
}

void MllmThreadPool::setAffinity(pthread_t handle, int cpu_id_mask) {
  // TODO
}

void MllmThreadPool::rebindCPUCore(size_t worker_id, int core_id_mask) {
  if (worker_id >= workers_.size()) {
    MLLM_ERROR_EXIT(kError, "Worker id {} is out of range.", worker_id);
  }

  auto handle = workers_[worker_id].native_handle();
  setAffinity(handle, core_id_mask);
}

}  // namespace mllm