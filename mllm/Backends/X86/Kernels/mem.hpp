/**
 * @file mem.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>

namespace mllm::X86 {

// aligned to 512bit(64B) vector.
void X86_align_alloc(void** ptr, size_t required_bytes, size_t align = 64);

void X86_align_free(void* ptr);

}  // namespace mllm::X86