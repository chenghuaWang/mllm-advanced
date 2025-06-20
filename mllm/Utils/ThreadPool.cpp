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
#include <sched.h>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>
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
  if (!initialized_) {
    MLLM_WARN("The thread pool has been initialized. Skip this initialization.");
  }

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
      system_workers_pid_.emplace_back(syscall(SYS_gettid));
      workLoop();
    });
  }

  initialized_ = true;
}

void MllmThreadPool::setAffinity(pid_t handle, int cpu_id_mask) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (int i = 0; i < sizeof(int) * 8; ++i) {
    if (cpu_id_mask & (1 << i)) { CPU_SET(i, &cpuset); }
  }

  int result = sched_setaffinity(handle, sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    MLLM_ERROR_EXIT(kError, "Failed to set thread affinity: error code {}. {}", result,
                    strerror(errno));
  }
}

int MllmThreadPool::getRunOnCPUCore(size_t worker_id) { return sched_getcpu(); }

void MllmThreadPool::rebindCPUCore(size_t worker_id, int core_id_mask) {
  if (worker_id >= workers_.size()) {
    MLLM_ERROR_EXIT(kError, "Worker id {} is out of range.", worker_id);
  }

  auto handle = system_workers_pid_[worker_id];
  setAffinity(handle, core_id_mask);
}

}  // namespace mllm