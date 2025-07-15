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

namespace {
inline void _ew_sub_fp32_tile_16(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vld1q_f32(b + offset);
    vst1q_f32(c + offset, vsubq_f32(va, vb));
  }
}

inline void _ew_mul_fp32_tile_16(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vld1q_f32(b + offset);
    vst1q_f32(c + offset, vmulq_f32(va, vb));
  }
}

inline void _ew_div_fp32_tile_16(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vld1q_f32(b + offset);
    vst1q_f32(c + offset, vdivq_f32(va, vb));
  }
}

inline void _ew_add_constant_fp32_tile_16(const float* __restrict__ a, const float b,
                                          float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vdupq_n_f32(b);
    vst1q_f32(c + offset, vaddq_f32(va, vb));
  }
}

inline void _ew_sub_constant_fp32_tile_16(const float* __restrict__ a, const float b,
                                          float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vdupq_n_f32(b);
    vst1q_f32(c + offset, vsubq_f32(va, vb));
  }
}

inline void _ew_mul_constant_fp32_tile_16(const float* __restrict__ a, const float b,
                                          float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vdupq_n_f32(b);
    vst1q_f32(c + offset, vmulq_f32(va, vb));
  }
}

inline void _ew_div_constant_fp32_tile_16(const float* __restrict__ a, const float b,
                                          float* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = i * 4;
    float32x4_t va = vld1q_f32(a + offset);
    float32x4_t vb = vdupq_n_f32(b);
    vst1q_f32(c + offset, vdivq_f32(va, vb));
  }
}
}  // namespace

void ew_add_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
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

void ew_sub_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

  if (threads) {
#pragma omp parallel for num_threads(threads) schedule(auto)
    for (int32_t b = 0; b < blocks; ++b) {
      const int32_t offset = b * TILE_SIZE;
      _ew_sub_fp32_tile_16(A + offset, B + offset, C + offset);
    }
  } else {
    for (int32_t b = 0; b < blocks; ++b) {
      const int32_t offset = b * TILE_SIZE;
      _ew_sub_fp32_tile_16(A + offset, B + offset, C + offset);
    }
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  const float* b_remain = B + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] - b_remain[i]; }
}

void ew_mul_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_mul_fp32_tile_16(A + offset, B + offset, C + offset);
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  const float* b_remain = B + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] * b_remain[i]; }
}

void ew_div_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_div_fp32_tile_16(A + offset, B + offset, C + offset);
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  const float* b_remain = B + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] / b_remain[i]; }
}

void ew_add_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_add_constant_fp32_tile_16(A + offset, B, C + offset);
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] + B; }
}

void ew_sub_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_sub_constant_fp32_tile_16(A + offset, B, C + offset);
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] - B; }
}

void ew_mul_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_mul_constant_fp32_tile_16(A + offset, B, C + offset);
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] * B; }
}

void ew_div_constant_fp32(const float* __restrict__ A, const float B, float* __restrict__ C,
                          int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 16;  // 4 vectors * 4 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_div_constant_fp32_tile_16(A + offset, B, C + offset);
  }

  const float* a_remain = A + blocks * TILE_SIZE;
  float* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] / B; }
}

void ew_neg_fp32(const float* __restrict__ A, float* __restrict__ B, int32_t len) {
  int32_t i = 0;
  int32_t simd_len = len & ~3;
  for (; i < simd_len; i += 4) {
    float32x4_t va = vld1q_f32(&A[i]);
    float32x4_t vb = vnegq_f32(va);
    vst1q_f32(&B[i], vb);
  }
  for (; i < len; ++i) { B[i] = -A[i]; }
}

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
void ew_add_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads) {
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

namespace {
inline void _ew_sub_fp16_tile_32(const float16_t* __restrict__ a, const float16_t* __restrict__ b,
                                 float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vld1q_f16(b + offset);
    vst1q_f16(c + offset, vsubq_f16(va, vb));
  }
}

inline void _ew_mul_fp16_tile_32(const float16_t* __restrict__ a, const float16_t* __restrict__ b,
                                 float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vld1q_f16(b + offset);
    vst1q_f16(c + offset, vmulq_f16(va, vb));
  }
}

inline void _ew_div_fp16_tile_32(const float16_t* __restrict__ a, const float16_t* __restrict__ b,
                                 float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vld1q_f16(b + offset);
    vst1q_f16(c + offset, vdivq_f16(va, vb));
  }
}

inline void _ew_add_constant_fp16_tile_32(const float16_t* __restrict__ a, const float16_t b,
                                          float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vdupq_n_f16(b);
    vst1q_f16(c + offset, vaddq_f16(va, vb));
  }
}

inline void _ew_sub_constant_fp16_tile_32(const float16_t* __restrict__ a, const float16_t b,
                                          float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vdupq_n_f16(b);
    vst1q_f16(c + offset, vsubq_f16(va, vb));
  }
}

inline void _ew_mul_constant_fp16_tile_32(const float16_t* __restrict__ a, const float16_t b,
                                          float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vdupq_n_f16(b);
    vst1q_f16(c + offset, vmulq_f16(va, vb));
  }
}

inline void _ew_div_constant_fp16_tile_32(const float16_t* __restrict__ a, const float16_t b,
                                          float16_t* __restrict__ c) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int offset = i * 8;
    float16x8_t va = vld1q_f16(a + offset);
    float16x8_t vb = vdupq_n_f16(b);
    vst1q_f16(c + offset, vdivq_f16(va, vb));
  }
}
}  // namespace

void ew_sub_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_sub_fp16_tile_32(A + offset, B + offset, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  const float16_t* b_remain = B + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] - b_remain[i]; }
}

void ew_mul_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_mul_fp16_tile_32(A + offset, B + offset, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  const float16_t* b_remain = B + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] * b_remain[i]; }
}

void ew_div_fp16(const float16_t* __restrict__ A, const float16_t* __restrict__ B,
                 float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_div_fp16_tile_32(A + offset, B + offset, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  const float16_t* b_remain = B + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] / b_remain[i]; }
}

void ew_add_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_add_constant_fp16_tile_32(A + offset, B, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] + B; }
}

void ew_sub_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_sub_constant_fp16_tile_32(A + offset, B, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] - B; }
}

void ew_mul_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_mul_constant_fp16_tile_32(A + offset, B, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] * B; }
}

void ew_div_constant_fp16(const float16_t* __restrict__ A, const float16_t B,
                          float16_t* __restrict__ C, int32_t len, int threads) {
  constexpr int32_t TILE_SIZE = 32;  // 4 vectors * 8 elements
  const int32_t blocks = len / TILE_SIZE;
  const int32_t lefts = len % TILE_SIZE;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (int32_t b = 0; b < blocks; ++b) {
    const int32_t offset = b * TILE_SIZE;
    _ew_div_constant_fp16_tile_32(A + offset, B, C + offset);
  }

  const float16_t* a_remain = A + blocks * TILE_SIZE;
  float16_t* c_remain = C + blocks * TILE_SIZE;
  for (int32_t i = 0; i < lefts; ++i) { c_remain[i] = a_remain[i] / B; }
}

void ew_neg_fp16(const float16_t* __restrict__ A, float16_t* __restrict__ B, int32_t len) {
  int32_t i = 0;
  int32_t simd_len = len & ~7;
  for (; i < simd_len; i += 8) {
    float16x8_t va = vld1q_f16(&A[i]);
    float16x8_t vb = vnegq_f16(va);
    vst1q_f16(&B[i], vb);
  }
  for (; i < len; ++i) { B[i] = -A[i]; }
}

#endif  // fp16

}  // namespace mllm::arm
#endif