/**
 * @file ArmAllocator.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/ArmAllocator.hpp"

namespace mllm::arm {

bool ArmAllocator::alloc(const std::shared_ptr<TensorImpl>& tensor) {
  void* ptr;
  arm_align_alloc(&ptr, tensor->size(), alignSize());
  if (!ptr) return false;
  tensor->_setRawPtr(ptr);
  return true;
}

void ArmAllocator::free(const std::shared_ptr<TensorImpl>& tensor) {
  arm_align_free(tensor->rptr());
}

void ArmAllocator::free(TensorImpl* tensor) { arm_align_free(tensor->rptr()); }

bool ArmAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  arm_align_alloc(ptr, cap, align);
  return ptr != nullptr;
}

void ArmAllocator::generalFree(void* ptr) {
  if (!ptr) return;
  arm_align_free(ptr);
}

size_t ArmAllocator::allocSize(const std::shared_ptr<TensorImpl>& tensor) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = tensor->size();
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t ArmAllocator::allocSize(TensorImpl* tensor) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = tensor->size();
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t ArmAllocator::alignSize() const {
#if __ARM_ARCH < 8
  return 16;
#else
  return 32;
#endif
}

}  // namespace mllm::arm
