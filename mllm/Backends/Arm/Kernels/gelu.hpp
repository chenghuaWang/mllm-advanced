/**
 * @file gelu.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <arm_neon.h>
#include <cstdint>

namespace mllm::arm {

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N);

void gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N);

void quick_gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N);

void quick_gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N);

}  // namespace mllm::arm
