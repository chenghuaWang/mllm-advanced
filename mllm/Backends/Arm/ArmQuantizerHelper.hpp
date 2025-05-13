/**
 * @file ArmQuantizerHelper.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <arm_neon.h>

namespace mllm::arm {

void pack_kxn_fp16_w_bias_kleidiai(float16_t* __restrict__ packed_weight,
                                   const float16_t* __restrict__ weight,
                                   const float16_t* __restrict__ bias, int K, int N);

}  // namespace mllm::arm