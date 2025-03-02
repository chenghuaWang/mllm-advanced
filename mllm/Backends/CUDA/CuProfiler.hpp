/**
 * @file CuProfiler.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <iostream>
#include <string>
#include <map>
#include <limits>
#include <cuda_runtime.h>

namespace mllm::cuda {

class CUDATimer {
 public:
  CUDATimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CUDATimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0) { cudaEventRecord(start_, stream); }

  void stop(cudaStream_t stream = 0) { cudaEventRecord(stop_, stream); }

  float elapsed() {
    cudaEventSynchronize(stop_);
    float ms;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_, stop_;
};

class CuProfiler {
 public:
  struct Record {
    std::string name;
    float total_time = 0;
    int count = 0;
    float max_time = 0;
    float min_time = std::numeric_limits<float>::max();
  };

  void addRecord(const std::string& name, float time) {
    auto& record = records_[name];
    record.name = name;
    record.total_time += time;
    record.count += 1;
    record.max_time = std::max(record.max_time, time);
    record.min_time = std::min(record.min_time, time);
  }

  void printReport() {
    std::cout << "CUDA Performance Report:\n";
    std::cout << "========================================\n";
    for (const auto& pair : records_) {
      const auto& record = pair.second;
      std::cout << "Operation: " << record.name << "\n"
                << "  Total Time:    " << record.total_time << " ms\n"
                << "  Calls:         " << record.count << "\n"
                << "  Average Time:  " << record.total_time / record.count << " ms\n"
                << "  Maximum Time:  " << record.max_time << " ms\n"
                << "  Minimum Time:  " << record.min_time << " ms\n"
                << "----------------------------------------\n";
    }
  }

 private:
  std::map<std::string, Record> records_;
};

class ProfileScope {
 public:
  ProfileScope(CuProfiler& profiler, const std::string& name, cudaStream_t stream = 0)
      : profiler_(profiler), name_(name), stream_(stream) {
    timer_.start(stream_);
  }

  ~ProfileScope() {
    timer_.stop(stream_);
    profiler_.addRecord(name_, timer_.elapsed());
  }

 private:
  CuProfiler& profiler_;
  std::string name_;
  CUDATimer timer_;
  cudaStream_t stream_;
};

#define MLLM_CUDA_PROFILE_FUNC(name, ...) \
  void MLLM_CUDA_PERF_##name(::mllm::cuda::CuProfiler& profiler, ##__VA_ARGS__)

}  // namespace mllm::cuda
