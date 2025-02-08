/**
 * @file softmax.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <arm_neon.h>

namespace mllm::arm {

// Safe sofmax for fp32. Not optimized for stride!=1 situation. When stride is set to 1, this
// function will utilize vexp1_fast_fp32 method to accelerate exp computation. This function not
// required (len % K == 0), any length is acceptable.
void softmax_V1(const float* __restrict X, float* __restrict Y, int len, int stride);

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else

void hsoftmax_V1(const float16_t* __restrict X, float16_t* __restrict Y, int len, int stride);

#endif  // fp16

}  // namespace mllm::arm

#endif
