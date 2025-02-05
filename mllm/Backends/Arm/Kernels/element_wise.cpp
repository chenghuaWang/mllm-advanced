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
#include "mllm/Utils/Log.hpp"
#include <omp.h>
#include <arm_neon.h>

namespace mllm::arm {

void ew_add_fp32(const float* __restrict A, const float* __restrict B, float* __restrict C,
                 int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 floats per SIMD operation * 4 operations

  if (threads) {
    int32_t thread_block_len = len / threads;
    int32_t thread_block_left = len % threads;

    if (thread_block_len < (1U << 12U)) {
      MLLM_WARN("thread_block_len < 4096 is inefficient. Pls use none threading version");
    }

#pragma omp parallel for num_threads(threads) schedule(static)
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
      auto a_thread_ptr = A + thread_id * thread_block_len;
      auto b_thread_ptr = B + thread_id * thread_block_len;
      auto c_thread_ptr = C + thread_id * thread_block_len;

      int32_t in_thread_blocks = thread_block_len / TILE_SIZE;
      int32_t in_thread_blocks_left = thread_block_len % TILE_SIZE;

      for (int b = 0; b < in_thread_blocks; ++b) {
        auto a_ptr = a_thread_ptr + b * TILE_SIZE;
        auto b_ptr = b_thread_ptr + b * TILE_SIZE;
        auto c_ptr = c_thread_ptr + b * TILE_SIZE;

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

      auto a_ptr = a_thread_ptr + in_thread_blocks * TILE_SIZE;
      auto b_ptr = b_thread_ptr + in_thread_blocks * TILE_SIZE;
      auto c_ptr = c_thread_ptr + in_thread_blocks * TILE_SIZE;

      for (int i = 0; i < in_thread_blocks_left; ++i) { c_ptr[i] = a_ptr[i] + b_ptr[i]; }
    }

    auto a_ptr = A + len - thread_block_left;
    auto b_ptr = B + len - thread_block_left;
    auto c_ptr = C + len - thread_block_left;

    for (int i = 0; i < thread_block_left; ++i) { c_ptr[i] = a_ptr[i] + b_ptr[i]; }

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

void ew_add_fp16(const float16_t* __restrict A, const float16_t* __restrict B,
                 float16_t* __restrict C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;

  if (threads) {
    int32_t thread_block_len = len / threads;
    int32_t thread_block_left = len % threads;

    if (thread_block_len < (1U << 12U)) {
      MLLM_WARN("thread_block_len < 4096 is inefficient. Pls use none threading version");
    }

#pragma omp parallel for num_threads(threads) schedule(static)
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
      auto a_thread_ptr = A + thread_id * thread_block_len;
      auto b_thread_ptr = B + thread_id * thread_block_len;
      auto c_thread_ptr = C + thread_id * thread_block_len;

      int32_t in_thread_blocks = thread_block_len / TILE_SIZE;
      int32_t in_thread_blocks_left = thread_block_len % TILE_SIZE;

      for (int b = 0; b < in_thread_blocks; ++b) {
        auto a_ptr = a_thread_ptr + b * TILE_SIZE;
        auto b_ptr = b_thread_ptr + b * TILE_SIZE;
        auto c_ptr = c_thread_ptr + b * TILE_SIZE;

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

      auto a_ptr = a_thread_ptr + in_thread_blocks * TILE_SIZE;
      auto b_ptr = b_thread_ptr + in_thread_blocks * TILE_SIZE;
      auto c_ptr = c_thread_ptr + in_thread_blocks * TILE_SIZE;

      for (int i = 0; i < in_thread_blocks_left; ++i) {
        c_ptr[i] = static_cast<float16_t>(a_ptr[i] + b_ptr[i]);
      }
    }

    auto a_ptr = A + len - thread_block_left;
    auto b_ptr = B + len - thread_block_left;
    auto c_ptr = C + len - thread_block_left;

    for (int i = 0; i < thread_block_left; ++i) {
      c_ptr[i] = static_cast<float16_t>(a_ptr[i] + b_ptr[i]);
    }

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

void ew_sub_fp32(const float* __restrict A, const float* __restrict B, float* __restrict C,
                 int32_t len, int threads) {
  // TODO
}

void ew_sub_fp16(const float16_t* __restrict A, const float16_t* __restrict B,
                 float16_t* __restrict C, int32_t len, int threads) {
  // TODO
}

}  // namespace mllm::arm
