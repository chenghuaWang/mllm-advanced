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

bool ArmAllocator::alloc(const std::shared_ptr<Storage>& storage) {
  void* ptr;
  arm_align_alloc(&ptr, storage->size_, alignSize());
  if (!ptr) return false;
  storage->ptr_ = ptr;
  return true;
}

void ArmAllocator::free(const std::shared_ptr<Storage>& storage) { arm_align_free(storage->ptr_); }

void ArmAllocator::free(Storage* storage) { arm_align_free(storage->ptr_); }

bool ArmAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  arm_align_alloc(ptr, cap, align);
  return ptr != nullptr;
}

void ArmAllocator::generalFree(void* ptr) {
  if (!ptr) return;
  arm_align_free(ptr);
}

size_t ArmAllocator::allocSize(Storage* storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t ArmAllocator::allocSize(const std::shared_ptr<Storage>& storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t ArmAllocator::alignSize() const {
#if __ARM_ARCH < 8
  return 16;
#else
  return 16;
#endif
}

}  // namespace mllm::arm
