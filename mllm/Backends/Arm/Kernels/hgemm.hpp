/**
 * @file hgemm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
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
void hgemm_mk_nk_mn_V1(const float16_t* __restrict lhs, const float16_t* __restrict rhs,
                       float16_t* __restrict dst, size_t M, size_t K, size_t N,
                       const float16_t* __restrict bias, int threads = 0);

void hgemm_mk_kn_mn_V1(const float16_t* __restrict lhs, const float16_t* __restrict rhs,
                       float16_t* __restrict dst, size_t M, size_t K, size_t N,
                       const float16_t* __restrict bias, int threads = 0);

}  // namespace mllm::arm

#endif
