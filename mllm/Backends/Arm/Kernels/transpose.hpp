/**
 * @file transpose.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>

namespace mllm::arm {

void transpose_bshd_bhsd(const float* __restrict X, float* __restrict Y, size_t B, size_t S,
                         size_t H, size_t D);

}
