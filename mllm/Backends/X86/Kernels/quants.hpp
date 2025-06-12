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
 * @file quants.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <half/half.hpp>
#include <cstring>
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

#define QK_K 256
#define MLLM_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MLLM_MAX(a, b) ((a) > (b) ? (a) : (b))

namespace mllm::X86 {

static inline int nearest_int(float fval) {
  MLLM_RT_ASSERT(fval <= 4194303.F);
  float val = fval + 12582912.F;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

static inline float make_qkx2_quants(int n, int nmax, const float* __restrict__ x,
                                     const float* __restrict__ weights, uint8_t* __restrict__ L,
                                     float* __restrict__ the_min, uint8_t* __restrict__ laux,
                                     float rmin, float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = -min;
    return 0.F;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MLLM_MAX(0, MLLM_MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0;
    float sum_l2 = 0;
    float sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MLLM_MAX(0, MLLM_MIN(nmax, l));
      laux[i] = l;
      float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) { L[i] = laux[i]; }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t* __restrict__ q, uint8_t* __restrict__ d,
                                    uint8_t* __restrict__ m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

void quantize_row_q4_k_reference(const float* __restrict__ x, block_q4_k_t* __restrict__ y, int k);

void quantize_row_q4_k(block_q4_k_t* __restrict__ vy, const float* __restrict__ x, int k);

void dequantize_row_q4_k(float* __restrict__ y, const block_q4_k_t* __restrict__ x, int k);

void quantize_row_q8_K_reference(const float* __restrict__ x, block_q8_k_t* __restrict__ y, int k);

void quantize_row_q8_k(void* __restrict__ y, const float* __restrict__ x, int k);

void dequantize_row_q8_k(float* __restrict__ y, const block_q8_k_t* __restrict__ x, int k);

}  // namespace mllm::X86
