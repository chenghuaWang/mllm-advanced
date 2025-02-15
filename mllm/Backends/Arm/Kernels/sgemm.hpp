/**
 * @file sgemm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
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

void sgemm_mk_nk_mn_V1(const float* __restrict lhs, const float* __restrict rhs,
                       float* __restrict dst, int M, int K, int N, const float* __restrict bias,
                       int threads = 0);

void sgemm_mk_kn_mn_V1(const float* __restrict lhs, const float* __restrict rhs,
                       float* __restrict dst, int M, int K, int N, const float* __restrict bias,
                       int threads = 0);

}  // namespace mllm::arm

#endif
