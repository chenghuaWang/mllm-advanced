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
#include "mllm/Backends/CUDA/Ops/D2HOp.hpp"
#include "mllm/Backends/CUDA/Ops/ElewiseOps.hpp"

namespace mllm::cuda {

CUDABackend::CUDABackend() : BackendBase(kCUDA) {
  GpuMetaInfo::instance();
  auto& devices = GpuMetaInfo::instance().devices;
  for (auto& d : devices) { MLLM_INFO("Found device: {}", d.name); }
  allocator_ = std::make_shared<CUDAAllocator>();
  regOpFactory<CUDAAddOpFactory, CUDAD2HOpFactory>();
}

std::shared_ptr<CUDABackend> createCUDABackend() { return std::make_shared<CUDABackend>(); }

}  // namespace mllm::cuda
