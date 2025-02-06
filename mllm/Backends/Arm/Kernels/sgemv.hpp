/**
 * @file sgemv.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#if !defined(__aarch64__)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
#include <arm_neon.h>

namespace mllm::arm {

void sgemv_1K_NK_V1(const float* __restrict A, const float* __restrict B,
                    const float* __restrict bias, float* __restrict C, int K, int N);

}

#endif