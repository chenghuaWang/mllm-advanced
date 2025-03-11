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
 * @file vec_dot_q4_k_q8_k_hw.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/X86/Kernels/vec_dot_q4_k_q8_k_avx2.hpp"
#include <half/half.hpp>

namespace mllm::X86 {

namespace {
__m256i get_scale_shuffle_k4(int i) {
  static const uint8_t KShuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return _mm256_loadu_si256((const __m256i*)KShuffle + i);
}

float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

}  // namespace

void vec_dot_q4_k_q8_k_avx2(float* C, const block_q4_k_t* __restrict__ A,
                            const __block_q8_k* __restrict__ B, const int num) {
  MLLM_RT_ASSERT(num % 256 == 0);

  const int nb = num / 256;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const __m256i m4 = _mm256_set1_epi8(0xF);

  __m256 acc = _mm256_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = B[i].d * half_float::half_cast<float>(A[i].d);
    const float dmin = -B[i].d * half_float::half_cast<float>(A[i].dmin);

    memcpy(utmp, A[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const uint8_t* __restrict__ q4 = A[i].qs;
    const int8_t* __restrict__ q8 = B[i].qs;

    const __m256i mins_and_scales =
        _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)B[i].bsums);
    const __m128i q8s =
        _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
    const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

    __m256i sumi = _mm256_setzero_si256();

#pragma unroll
    for (int j = 0; j < 256 / 64; ++j) {
      const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
      const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

      const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
      q4 += 32;
      const __m256i q4l = _mm256_and_si256(q4bits, m4);
      const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

      const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
      p16l = _mm256_madd_epi16(scale_l, p16l);

      const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
      p16h = _mm256_madd_epi16(scale_h, p16h);
      const __m256i sumj = _mm256_add_epi32(p16l, p16h);

      sumi = _mm256_add_epi32(sumi, sumj);
    }

    __m256 vd = _mm256_set1_ps(d);
    acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
  }

  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

  *C = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

}  // namespace mllm::X86
