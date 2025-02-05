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

namespace mllm::arm {

// aligned to 128bit(16B) vector.
void arm_align_alloc(void** ptr, size_t required_bytes, size_t align = 16);

void arm_align_free(void* ptr);

}  // namespace mllm::arm
