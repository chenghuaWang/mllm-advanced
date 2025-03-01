/**
 * @file CUDABackend.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/CUDABackend.hpp"
#include "mllm/Backends/CUDA/CUDAAllocator.hpp"
#include "mllm/Backends/CUDA/CUDACommons.hpp"
#include "mllm/Backends/CUDA/Ops/D2HOp.cuh"
#include "mllm/Backends/CUDA/Ops/ElewiseOps.cuh"

namespace mllm::cuda {

CUDABackend::CUDABackend() : BackendBase(kCUDA) {
  GpuMetaInfo::instance();
  allocator_ = std::make_shared<CUDAAllocator>();
  regOpFactory<CUDAAddOpFactory, CUDAD2HOpFactory>();
}

std::shared_ptr<CUDABackend> createCUDABackend() { return std::make_shared<CUDABackend>(); }

}  // namespace mllm::cuda
