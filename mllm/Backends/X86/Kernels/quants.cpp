/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/**
 * @file quants.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/X86/Kernels/quants.hpp"
#include "half/half.hpp"

namespace mllm::X86 {

void dequantize_row_q4_k(float* __restrict y, const block_q4_k_t* __restrict x, int k) {
  MLLM_RT_ASSERT_EQ(k % QK_K, 0);

  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t* q = x[i].qs;

#if QK_K == 256
    const float d = half_float::half_cast<float>(x[i].d);
    const float min = half_float::half_cast<float>(x[i].dmin);
    int is = 0;
    uint8_t sc;
    uint8_t m;

#pragma unroll
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
#else
    const float dall = half_float::half_cast<float>(x[i].d[0]);
    const float mall = half_float::half_cast<float>(x[i].d[1]);
    const float d1 = dall * (x[i].scales[0] & 0xF), m1 = mall * (x[i].scales[0] >> 4);
    const float d2 = dall * (x[i].scales[1] & 0xF), m2 = mall * (x[i].scales[1] >> 4);
    for (int l = 0; l < 32; ++l) {
      y[l + 0] = d1 * (q[l] & 0xF) - m1;
      y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
    y += QK_K;
#endif
  }
}

void quantize_row_q4_k_reference(const float* __restrict x, block_q4_k_t* __restrict y, int k) {
  MLLM_RT_ASSERT(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float weights[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.F,
                                   0.1F, 20, false);
      float scale = scales[j];
      if (scale > max_scale) { max_scale = scale; }
      float min = mins[j];
      if (min > max_min) { max_min = min; }
    }

#if QK_K == 256
    float inv_scale = max_scale > 0 ? 63.F / max_scale : 0.F;
    float inv_min = max_min > 0 ? 63.F / max_min : 0.F;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MLLM_MIN(63, ls);
      lm = MLLM_MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = half_float::half(max_scale / 63.F);
    y[i].dmin = half_float::half(max_min / 63.F);

    uint8_t sc;
    uint8_t m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = half_float::half_cast<float>(y[i].d) * sc;
      if (d == 0.0F) continue;
      const float dm = half_float::half_cast<float>(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MLLM_MAX(0, MLLM_MIN(15, l));
        L[32 * j + ii] = l;
      }
    }
#else
    const float s_factor = 15.f;
    float inv_scale = max_scale > 0 ? s_factor / max_scale : 0.f;
    float inv_min = max_min > 0 ? s_factor / max_min : 0.f;
    int d1 = nearest_int(inv_scale * scales[0]);
    int m1 = nearest_int(inv_min * mins[0]);
    int d2 = nearest_int(inv_scale * scales[1]);
    int m2 = nearest_int(inv_min * mins[1]);
    y[i].scales[0] = d1 | (m1 << 4);
    y[i].scales[1] = d2 | (m2 << 4);
    y[i].d[0] = half_float::half(max_scale / s_factor);
    y[i].d[1] = half_float::half(max_min / s_factor);

    float sumlx = 0;
    int suml2 = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      const uint8_t sd = y[i].scales[j] & 0xF;
      const uint8_t sm = y[i].scales[j] >> 4;
      const float d = half_float::half_cast<float>(y[i].d[0]) * sd;
      if (!d) continue;
      const float m = half_float::half_cast<float>(y[i].d[1]) * sm;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + m) / d);
        l = MLLM_MAX(0, MLLM_MIN(15, l));
        L[32 * j + ii] = l;
        sumlx += (x[32 * j + ii] + m) * l * sd;
        suml2 += l * l * sd * sd;
      }
    }
    if (suml2) { y[i].d[0] = half_float::half(sumlx / suml2); }
#endif
    uint8_t* q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

void quantize_row_q4_k(block_q4_k_t* __restrict vy, const float* __restrict x, int k) {
  MLLM_RT_ASSERT(k % QK_K == 0);
  block_q4_k_t* __restrict y = (block_q4_k_t*)vy;
  quantize_row_q4_k_reference(x, y, k);
}

void quantize_row_q8_K_reference(const float* __restrict x, block_q8_k_t* __restrict y, int k) {
  MLLM_RT_ASSERT(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (amax == 0.0F) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    const float iscale = -128.F / max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = MLLM_MIN(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) { sum += y[i].qs[j * 16 + ii]; }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1 / iscale;
    x += QK_K;
  }
}

void quantize_row_q8_k(void* __restrict y, const float* __restrict x, int k) {
  quantize_row_q8_K_reference(x, (block_q8_k_t*)y, k);
}

void dequantize_row_q8_k(float* __restrict y, const block_q8_k_t* __restrict x, int k) {
  MLLM_RT_ASSERT(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K; ++j) { *y++ = x[i].d * x[i].qs[j]; }
  }
}

}  // namespace mllm::X86
