/**
 * @file X86Allocator.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/X86/Kernels/mem.hpp"
#include "mllm/Backends/X86/X86Allocator.hpp"

namespace mllm::X86 {

bool X86Allocator::alloc(const std::shared_ptr<TensorImpl>& tensor) {
  void* ptr;
  X86_align_alloc(&ptr, tensor->size(), alignSize());
  if (!ptr) return false;
  tensor->_setRawPtr(ptr);
  return true;
}

void X86Allocator::free(const std::shared_ptr<TensorImpl>& tensor) {
  X86_align_free(tensor->rptr());
}

void X86Allocator::free(TensorImpl* tensor) { X86_align_free(tensor->rptr()); }

bool X86Allocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  X86_align_alloc(ptr, cap, align);
  return ptr != nullptr;
}

void X86Allocator::generalFree(void* ptr) {
  if (!ptr) return;
  X86_align_free(ptr);
}

size_t X86Allocator::allocSize(const std::shared_ptr<TensorImpl>& tensor) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = tensor->size();
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t X86Allocator::allocSize(TensorImpl* tensor) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = tensor->size();
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t X86Allocator::alignSize() const { return 64; }

}  // namespace mllm::X86
