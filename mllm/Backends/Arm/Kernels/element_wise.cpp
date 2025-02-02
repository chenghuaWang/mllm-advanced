/**
 * @file element_wise.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/element_wise.hpp"
#include <omp.h>
#include <arm_neon.h>

namespace mllm::arm {

void ew_add_fp32(const float* __restrict A, const float* __restrict B, float* __restrict C,
                 int32_t len, int threads) {
  // TODO
}

void ew_add_fp16(const float16_t* __restrict A, const float16_t* __restrict B,
                 float16_t* __restrict C, int32_t len, int threads) {
  // TODO
}

}  // namespace mllm::arm
