/**
 * @file mem.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstdlib>
#include "mllm/Backends/X86/Kernels/mem.hpp"

namespace mllm::X86 {
void X86_align_alloc(void** ptr, size_t required_bytes, size_t align) {
  if (align == 0 || (align & (align - 1))) {
    *ptr = nullptr;
    return;
  }
  void* p1;
  void** p2;
  size_t offset = align - 1 + sizeof(void*);
  if ((p1 = (void*)malloc(required_bytes + offset)) == nullptr) {
    *ptr = nullptr;
    return;
  }
  p2 = (void**)(((size_t)(p1) + offset) & ~(align - 1));
  p2[-1] = p1;
  *ptr = p2;
}

void X86_align_free(void* ptr) { free(((void**)ptr)[-1]); }
}  // namespace mllm::X86
