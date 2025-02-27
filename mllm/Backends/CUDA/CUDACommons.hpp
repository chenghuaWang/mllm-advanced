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

#include <cuda_runtime.h>
#include "mllm/Utils/Common.hpp"

#define MLLM_CHECK_CUDA_ERROR(err)                                          \
  if (err != cudaSuccess) {                                                 \
    MLLM_ERROR_EXIT(kCudaError, "CUDA error: {}", cudaGetErrorString(err)); \
  }

namespace mllm::cuda {}
