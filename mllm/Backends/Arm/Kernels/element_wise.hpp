/**
 * @file element_wise.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <arm_neon.h>
#include <cstdint>

namespace mllm::arm {
void ew_add_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads = 0);

void ew_sub_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads = 0);

void ew_mul_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads = 0);

void ew_div_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads = 0);

void ew_add_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads = 0);

void ew_sub_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads = 0);

void ew_mul_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads = 0);

void ew_div_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads = 0);

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
void ew_add_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_sub_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_mul_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_div_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_add_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_sub_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_mul_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads = 0);

void ew_div_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads = 0);
#endif  // fp16
}  // namespace mllm::arm

#endif