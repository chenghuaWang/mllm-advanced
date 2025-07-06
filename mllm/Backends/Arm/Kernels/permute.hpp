/**
 * @file permute.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else
#include <arm_neon.h>
#include <cstddef>

namespace mllm::arm {

void permute_fp32(const float* __restrict__ input, float* __restrict__ output,
                  const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);

void permute_fp16(const float16_t* __restrict__ input, float16_t* __restrict__ output,
                  const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);

}  // namespace mllm::arm

#endif