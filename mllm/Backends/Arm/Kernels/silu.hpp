/**
 * @file silu.hpp
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

void silu_V1(const float* __restrict X, float* __restrict Y, int len);

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else

void silu_fp16_V1(const float16_t* __restrict X, float16_t* __restrict Y, int len);

#endif  // fp16

}  // namespace mllm::arm

#endif