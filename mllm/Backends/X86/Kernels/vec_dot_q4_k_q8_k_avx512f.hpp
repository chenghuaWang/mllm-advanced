/**
 * @file vec_dot_q4_k_q8_k_avx512f.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/DataTypes.hpp"

#if !defined(__AVX512F__)
#error The avx512 is required to compile this file.
#else

namespace mllm::X86 {

void vec_dot_q4_k_q8_k_avx512f(float* C, const block_q4_k_t* __restrict__ A,
                               const __block_q8_k* __restrict__ B, const int num);

}  // namespace mllm::X86

#endif
