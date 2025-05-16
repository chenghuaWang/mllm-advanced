/**
 * @file ThreadPool.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <sched.h>
#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <cstddef>

namespace mllm {

class MllmThreadPool {
  int num_threads_;
  std::vector<std::thread> workers_;
  std::vector<pid_t> system_workers_pid_;
  std::queue<std::function<void()>> queue_tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  std::atomic_bool stop_;

  /// Mark thread local
  static thread_local int current_thread_id_;

 public:
  static MllmThreadPool& instance() {
    static MllmThreadPool instance;
    return instance;
  }

  ~MllmThreadPool();

  MllmThreadPool() = default;

  void initialize(int num_threads);

  void setAffinity(pid_t handle, int cpu_id_mask);

  int getRunOnCPUCore(size_t worker_id);

  void rebindCPUCore(size_t worker_id, int core_id_mask);

  inline int getCurrentThreadId() { return current_thread_id_; }

  template<typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
    using return_type = typename std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::lock_guard lock(queue_mutex_);
      queue_tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return res;
  }

  // Auto chunked
  template<typename Func>
  void parallelFor(int start, int end, int step, Func func) {
    int adjusted_end = end;
    int num_iters = (adjusted_end - start + step - 1) / step;
    int num_chunks = num_threads_;
    int chunk_size = num_iters / num_chunks;
    int remainder = num_iters % num_chunks;

    std::vector<std::future<void>> futures;
    int current = start;
    for (int i = 0; i < num_chunks; ++i) {
      int iters = chunk_size + (i < remainder ? 1 : 0);
      if (iters == 0) break;
      int chunk_end = current + iters * step;
      if (chunk_end > end) chunk_end = end;
      futures.emplace_back(enqueue([=, &func]() {
        for (int j = current; j < chunk_end; j += step) { func(j); }
      }));
      current = chunk_end;
    }
    for (auto& fut : futures) fut.wait();
  }

  template<typename Func>
  void parallelForWoChunk(int start, int end, int step, Func func) {
    std::vector<std::future<void>> futures;
    int current = start;
    while (current < end) {
      int chunk_end = current + step;
      if (chunk_end > end) chunk_end = end;
      futures.emplace_back(enqueue([=, &func]() {
        for (int j = current; j < chunk_end; j += step) { func(j); }
      }));
      current = chunk_end;
    }
    for (auto& fut : futures) fut.wait();
  }

 private:
  void workLoop() {
    while (!stop_.load(std::memory_order_acquire)) {
      std::function<void()> task;
      {
        std::unique_lock lock(queue_mutex_);
        cv_.wait(lock,
                 [this] { return stop_.load(std::memory_order_relaxed) || !queue_tasks_.empty(); });

        if (stop_ && queue_tasks_.empty()) return;
        if (queue_tasks_.empty()) continue;

        task = std::move(queue_tasks_.front());
        queue_tasks_.pop();
      }
      task();
    }
  }
};

#define MLLM_THREAD_POOL_INIT(num_threads) MllmThreadPool::instance().initialize(num_threads)

#define MLLM_THIS_THREAD_ID MllmThreadPool::instance().getCurrentThreadId()

#define MLLM_PARALLEL_FOR(var, start, end) \
    MllmThreadPool::instance().parallelFor(start, end, 1, [&](int var)

#define MLLM_PARALLEL_FOR_STEP(var, start, end, step) \
    MllmThreadPool::instance().parallelFor(start, end, step, [&](int var)

#define MLLM_PARALLEL_FOR_CHUNK(var, start, end, step) \
    MllmThreadPool::instance().parallelForWoChunk(start, end, step, [&](int var)

#define MLLM_PARALLEL_FOR_END );

#define MLLM_BIND_CURRENT_THREAD(mask) \
  MllmThreadPool::instance().rebindCPUCore(MLLM_THIS_THREAD_ID, mask)

#define MLLM_BIND_WORKER(worker_id, mask) MllmThreadPool::instance().rebindCPUCore(worker_id, mask)

#define MLLM_CUR_RUN_ON_CPU_ID MllmThreadPool::instance().getRunOnCPUCore(MLLM_THIS_THREAD_ID)

}  // namespace mllm
