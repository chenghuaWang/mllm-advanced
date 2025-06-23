/**
 * @file fa2_mma0.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <arm_neon.h>

namespace mllm::arm {

/**
 * @brief Flash Attention 2's MMA0 Kernel. Impl in assembly code.
 *
 * @note dtype_t is float16_t and acc_dtype_t is float32_t
 *
 */
extern "C" void fa2_mma0_bshd_fp16_br4_bc4_neon_asm_micro_kernel(
    const float16_t* __restrict__ q_block, const float16_t* __restrict__ k_block,
    float* __restrict__ acc_s, const int32_t dim_size, const int32_t stride_q,
    const int32_t stride_k, const int32_t stride_acc);

}  // namespace mllm::arm
