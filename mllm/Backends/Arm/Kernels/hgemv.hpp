/**
 * @file hgemv.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
#include <arm_neon.h>

namespace mllm::arm {

// @chenghuaWang
//
// Optimized for:
// 1. Armv8.2-a with FP16 support
// 2. Cacheline size 64 bytes
// 3. N and K is power of 2
//
// This Half Precision GEMV is for nn::Linear in decoding stage of LLM.
void hgemv_1K_NK_V1(const float16_t* __restrict A, const float16_t* __restrict B,
                    const float16_t* __restrict bias, float16_t* __restrict C, int K, int N);

// High Presion FP16 GEMV
//
// !!! The precision of V2 is one order of magnitude higher than that of V1.
// !!! But V1 is 1.3 times faster than V2.
//
// Optimized for:
// 1. Armv8.2-a with FP16 support
// 2. Cacheline size 64 bytes
// 3. N and K is power of 2
//
// This Half Precision GEMV is for nn::Linear in decoding stage of LLM.
void hgemv_1K_NK_V2_HP(const float16_t* __restrict A, const float16_t* __restrict B,
                       const float16_t* __restrict bias, float16_t* __restrict C, int K, int N);
}  // namespace mllm::arm

#endif