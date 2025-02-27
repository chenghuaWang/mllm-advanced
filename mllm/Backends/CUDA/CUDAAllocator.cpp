/**
 * @file CUDAAllocator.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/CUDAAllocator.hpp"
#include "mllm/Backends/CUDA/CUDACommons.hpp"

namespace mllm::cuda {

bool CUDAAllocator::alloc(const std::shared_ptr<TensorImpl>& tensor) {}

void CUDAAllocator::free(const std::shared_ptr<TensorImpl>& tensor) {}

void CUDAAllocator::free(TensorImpl* tensor) {}

bool CUDAAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  size_t aligned_cap = ((cap + align - 1) / align) * align;

  // Allocate memory on the device
  MLLM_CHECK_CUDA_ERROR(cudaMalloc(ptr, aligned_cap));

  return true;
}

void CUDAAllocator::generalFree(void* ptr) { MLLM_CHECK_CUDA_ERROR(cudaFree(ptr)); }

size_t CUDAAllocator::allocSize(const std::shared_ptr<TensorImpl>& tensor) {}

size_t CUDAAllocator::allocSize(TensorImpl* tensor) {}

size_t CUDAAllocator::alignSize() const {}

}  // namespace mllm::cuda
