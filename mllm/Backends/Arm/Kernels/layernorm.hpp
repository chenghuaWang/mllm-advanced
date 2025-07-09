/**
 * @file layernorm.hpp
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

void layernorm_N_fp32(float* __restrict__ Z, const float* __restrict__ X,
                      const float* __restrict__ gamma, const float* __restrict__ beta, size_t N,
                      float eps);

void layernorm_N_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X,
                      const float16_t* __restrict__ gamma, const float16_t* __restrict__ beta,
                      size_t N, float eps);

}  // namespace mllm::arm
