/**
 * @file CUDACommons.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <nvml.h>
#include <cuda_runtime.h>
#include "mllm/Utils/Common.hpp"

#define MLLM_CHECK_CUDA_ERROR(err)                                          \
  if (err != cudaSuccess) {                                                 \
    MLLM_ERROR_EXIT(kCudaError, "CUDA error: {}", cudaGetErrorString(err)); \
  }

#define MLLM_CHECK_NVML_ERROR(err) \
  if (err != NVML_SUCCESS) { MLLM_ERROR_EXIT(kCudaError, "NVML error: {}", nvmlErrorString(err)); }

namespace mllm::cuda {

struct GpuInfo {
  std::string name;
  unsigned int id;
  unsigned int sm_count;
  unsigned int cuda_core_per_sm;
  unsigned int tensor_core_per_sm;
  unsigned int l1_cache;           // bytes
  unsigned int shared_mem_per_sm;  // bytes
  unsigned int global_mem;         // bytes
  unsigned int max_thread_per_sm;
  unsigned int warp_size_in_thread;
  unsigned int architecture;
};

class GpuMetaInfo {
 public:
  GpuMetaInfo();

  static GpuMetaInfo& instance() {
    static GpuMetaInfo instance;
    return instance;
  }

  GpuMetaInfo(const GpuMetaInfo&) = delete;
  GpuMetaInfo& operator=(const GpuMetaInfo&) = delete;
};

}  // namespace mllm::cuda
