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
#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else
#include <arm_neon.h>
#include <cstddef>

namespace mllm::arm {

void transpose_hw_wh(const float* __restrict X, float* __restrict Y, size_t H, size_t W);

void transpose_bshd_bhsd(const float* __restrict X, float* __restrict Y, size_t B, size_t S,
                         size_t H, size_t D);

void transpose_hw_wh_fp16(const float16_t* __restrict X, float16_t* __restrict Y, size_t H, size_t W);

void transpose_bshd_bhsd_fp16(const float16_t* __restrict X, float16_t* __restrict Y, size_t B,
                              size_t S, size_t H, size_t D);

}  // namespace mllm::arm
#endif