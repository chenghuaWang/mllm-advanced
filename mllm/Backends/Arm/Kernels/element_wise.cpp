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
#if !defined(__aarch64__)
#error Arm compiler is required.
#else
#include <arm_neon.h>

namespace mllm::arm {

void ew_add_fp32(const float* __restrict A, const float* __restrict B, float* __restrict C,
                 int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 floats per SIMD operation * 4 operations

  if (threads) {
    int blocks = len / TILE_SIZE;
    int lefts = len % TILE_SIZE;
#pragma omp parallel for num_threads(threads) schedule(auto)
    for (int b = 0; b < blocks; b++) {
      auto a_ptr = A + b * TILE_SIZE;
      auto b_ptr = B + b * TILE_SIZE;
      auto c_ptr = C + b * TILE_SIZE;

      float32x4_t a_vec_0 = vld1q_f32(a_ptr);
      float32x4_t b_vec_0 = vld1q_f32(b_ptr);
      float32x4_t c_vec_0 = vaddq_f32(a_vec_0, b_vec_0);
      vst1q_f32(c_ptr, c_vec_0);

      float32x4_t a_vec_1 = vld1q_f32(a_ptr + 4);
      float32x4_t b_vec_1 = vld1q_f32(b_ptr + 4);
      float32x4_t c_vec_1 = vaddq_f32(a_vec_1, b_vec_1);
      vst1q_f32(c_ptr + 4, c_vec_1);

      float32x4_t a_vec_2 = vld1q_f32(a_ptr + 8);
      float32x4_t b_vec_2 = vld1q_f32(b_ptr + 8);
      float32x4_t c_vec_2 = vaddq_f32(a_vec_2, b_vec_2);
      vst1q_f32(c_ptr + 8, c_vec_2);

      float32x4_t a_vec_3 = vld1q_f32(a_ptr + 12);
      float32x4_t b_vec_3 = vld1q_f32(b_ptr + 12);
      float32x4_t c_vec_3 = vaddq_f32(a_vec_3, b_vec_3);
      vst1q_f32(c_ptr + 12, c_vec_3);
    }

    auto a_ptr = A + blocks * TILE_SIZE;
    auto b_ptr = B + blocks * TILE_SIZE;
    auto c_ptr = C + blocks * TILE_SIZE;

    for (int i = 0; i < lefts; ++i) { c_ptr[i] = a_ptr[i] + b_ptr[i]; }

    return;
  }

  int blocks = len / TILE_SIZE;
  int lefts = len % TILE_SIZE;

  for (int b = 0; b < blocks; b++) {
    auto a_ptr = A + b * TILE_SIZE;
    auto b_ptr = B + b * TILE_SIZE;
    auto c_ptr = C + b * TILE_SIZE;

    float32x4_t a_vec_0 = vld1q_f32(a_ptr);
    float32x4_t b_vec_0 = vld1q_f32(b_ptr);
    float32x4_t c_vec_0 = vaddq_f32(a_vec_0, b_vec_0);
    vst1q_f32(c_ptr, c_vec_0);

    float32x4_t a_vec_1 = vld1q_f32(a_ptr + 4);
    float32x4_t b_vec_1 = vld1q_f32(b_ptr + 4);
    float32x4_t c_vec_1 = vaddq_f32(a_vec_1, b_vec_1);
    vst1q_f32(c_ptr + 4, c_vec_1);

    float32x4_t a_vec_2 = vld1q_f32(a_ptr + 8);
    float32x4_t b_vec_2 = vld1q_f32(b_ptr + 8);
    float32x4_t c_vec_2 = vaddq_f32(a_vec_2, b_vec_2);
    vst1q_f32(c_ptr + 8, c_vec_2);

    float32x4_t a_vec_3 = vld1q_f32(a_ptr + 12);
    float32x4_t b_vec_3 = vld1q_f32(b_ptr + 12);
    float32x4_t c_vec_3 = vaddq_f32(a_vec_3, b_vec_3);
    vst1q_f32(c_ptr + 12, c_vec_3);
  }

  auto a_ptr = A + blocks * TILE_SIZE;
  auto b_ptr = B + blocks * TILE_SIZE;
  auto c_ptr = C + blocks * TILE_SIZE;

  for (int i = 0; i < lefts; ++i) { c_ptr[i] = a_ptr[i] + b_ptr[i]; }
}

void ew_sub_fp32(const float* __restrict A, const float* __restrict B, float* __restrict C,
                 int32_t len, int threads) {
  // TODO
}

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
void ew_add_fp16(const float16_t* __restrict A, const float16_t* __restrict B,
                 float16_t* __restrict C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;

  if (threads) {
    int blocks = len / TILE_SIZE;
    int lefts = len % TILE_SIZE;
#pragma omp parallel for num_threads(threads) schedule(auto)
    for (int b = 0; b < blocks; b++) {
      auto a_ptr = A + b * TILE_SIZE;
      auto b_ptr = B + b * TILE_SIZE;
      auto c_ptr = C + b * TILE_SIZE;

      float16x8_t a_vec_0 = vld1q_f16(a_ptr);
      float16x8_t b_vec_0 = vld1q_f16(b_ptr);
      float16x8_t c_vec_0 = vaddq_f16(a_vec_0, b_vec_0);
      vst1q_f16(c_ptr, c_vec_0);

      float16x8_t a_vec_1 = vld1q_f16(a_ptr + 8);
      float16x8_t b_vec_1 = vld1q_f16(b_ptr + 8);
      float16x8_t c_vec_1 = vaddq_f16(a_vec_1, b_vec_1);
      vst1q_f16(c_ptr + 8, c_vec_1);

      float16x8_t a_vec_2 = vld1q_f16(a_ptr + 16);
      float16x8_t b_vec_2 = vld1q_f16(b_ptr + 16);
      float16x8_t c_vec_2 = vaddq_f16(a_vec_2, b_vec_2);
      vst1q_f16(c_ptr + 16, c_vec_2);

      float16x8_t a_vec_3 = vld1q_f16(a_ptr + 24);
      float16x8_t b_vec_3 = vld1q_f16(b_ptr + 24);
      float16x8_t c_vec_3 = vaddq_f16(a_vec_3, b_vec_3);
      vst1q_f16(c_ptr + 24, c_vec_3);
    }

    auto a_ptr = A + blocks * TILE_SIZE;
    auto b_ptr = B + blocks * TILE_SIZE;
    auto c_ptr = C + blocks * TILE_SIZE;

    for (int i = 0; i < lefts; ++i) { c_ptr[i] = static_cast<float16_t>(a_ptr[i] + b_ptr[i]); }
    return;
  }

  int blocks = len / TILE_SIZE;
  int lefts = len % TILE_SIZE;

  for (int b = 0; b < blocks; b++) {
    auto a_ptr = A + b * TILE_SIZE;
    auto b_ptr = B + b * TILE_SIZE;
    auto c_ptr = C + b * TILE_SIZE;

    float16x8_t a_vec_0 = vld1q_f16(a_ptr);
    float16x8_t b_vec_0 = vld1q_f16(b_ptr);
    float16x8_t c_vec_0 = vaddq_f16(a_vec_0, b_vec_0);
    vst1q_f16(c_ptr, c_vec_0);

    float16x8_t a_vec_1 = vld1q_f16(a_ptr + 8);
    float16x8_t b_vec_1 = vld1q_f16(b_ptr + 8);
    float16x8_t c_vec_1 = vaddq_f16(a_vec_1, b_vec_1);
    vst1q_f16(c_ptr + 8, c_vec_1);

    float16x8_t a_vec_2 = vld1q_f16(a_ptr + 16);
    float16x8_t b_vec_2 = vld1q_f16(b_ptr + 16);
    float16x8_t c_vec_2 = vaddq_f16(a_vec_2, b_vec_2);
    vst1q_f16(c_ptr + 16, c_vec_2);

    float16x8_t a_vec_3 = vld1q_f16(a_ptr + 24);
    float16x8_t b_vec_3 = vld1q_f16(b_ptr + 24);
    float16x8_t c_vec_3 = vaddq_f16(a_vec_3, b_vec_3);
    vst1q_f16(c_ptr + 24, c_vec_3);
  }

  auto a_ptr = A + blocks * TILE_SIZE;
  auto b_ptr = B + blocks * TILE_SIZE;
  auto c_ptr = C + blocks * TILE_SIZE;

  for (int i = 0; i < lefts; ++i) { c_ptr[i] = static_cast<float16_t>(a_ptr[i] + b_ptr[i]); }
}

void ew_sub_fp16(const float16_t* __restrict A, const float16_t* __restrict B,
                 float16_t* __restrict C, int32_t len, int threads) {
  // TODO
}
#endif  // fp16

}  // namespace mllm::arm
#endif