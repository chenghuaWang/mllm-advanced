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

bool CUDAAllocator::alloc(const std::shared_ptr<Storage>& storage) {
  void* ptr;
  generalAlloc(&ptr, storage->size_, alignSize());
  if (!ptr) return false;
  storage->ptr_ = ptr;
  return true;
}

void CUDAAllocator::free(const std::shared_ptr<Storage>& storage) { generalFree(storage->ptr_); }

void CUDAAllocator::free(Storage* storage) { generalFree(storage->ptr_); }

bool CUDAAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  size_t aligned_cap = (cap + align - 1) & ~(align - 1);

  // Allocate memory on the device
  MLLM_CHECK_CUDA_ERROR(cudaMalloc(ptr, aligned_cap));

  return true;
}

void CUDAAllocator::generalFree(void* ptr) {
  MLLM_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  MLLM_CHECK_CUDA_ERROR(cudaFree(ptr));
}

size_t CUDAAllocator::allocSize(const std::shared_ptr<Storage>& storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t CUDAAllocator::allocSize(Storage* storage) {
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t CUDAAllocator::alignSize() const { return 32; }

}  // namespace mllm::cuda
