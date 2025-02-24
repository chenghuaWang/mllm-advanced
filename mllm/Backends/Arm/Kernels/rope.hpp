/**
 * @file rope.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-10
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

// Support Models:
// > Qwen2
// > Llama2
void precompute_normal_hf_sin_cos(int seq_len, int output_dim, float base, float* __restrict sin,
                                  float* __restrict cos, int threads = 0);

// Support Models:
// > Qwen2
// > Llama2
//
// Data Layout: [..., seq, dim]
void normal_hf_rope(const float* __restrict X, float* Y, const float* __restrict sin,
                    const float* __restrict cos, int cur_seq_cnt, int seq, int dims,
                    int threads = 0);

// Support Models:
// > Qwen2
// > Llama2
//
// Data Layout: [..., seq, dim]
void normal_hf_rope_fp16(const float16_t* __restrict X, float16_t* Y, const float* __restrict sin,
                         const float* __restrict cos, int cur_seq_cnt, int seq, int dims,
                         int threads = 0);
}  // namespace mllm::arm

#endif
