/**
 * @file vec_dot_q4_k_q8_k_avx512f.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <immintrin.h>
#include <half/half.hpp>
#include "mllm/Utils/Common.hpp"

#if defined(__AVX512F__)

#include "mllm/Backends/X86/Kernels/vec_dot_q4_k_q8_k_avx512f.hpp"

namespace mllm::X86 {

namespace {
__m512i get_scale_shuffle_avx512(int i) {
  static const uint8_t k_shuffle[256] = {
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
  return _mm512_loadu_si512((const __m512i*)k_shuffle + i);
}
}  // namespace

void vec_dot_q4_k_q8_k_avx512f(float* C, const block_q4_k_t* __restrict__ A,
                               const __block_q8_k* __restrict__ B, const int num) {
  MLLM_RT_ASSERT_EQ(num % 256, 0);
  const int nb = num / 256;

  static const uint32_t kmask1 = 0x3f3f3f3f;  // 00111111
  static const uint32_t kmask2 = 0x0f0f0f0f;  // 00001111
  static const uint32_t kmask3 = 0x03030303;  // 00000011

  uint32_t utmp[4];

  const __m512i m4 = _mm512_set1_epi8(0xF);  // 1111
  __m512 acc = _mm512_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = B[i].d * half_float::half_cast<float>(A[i].d);         // d_x * d_y
    const float dmin = -B[i].d * half_float::half_cast<float>(A[i].dmin);  // -b_x * d_y

    // combine 4 x 6 bits in 8 x fp32
    // group 0.
    memcpy(utmp, A[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const __m256i mins_and_scales =
        _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

    //  negative ReduceSUM(S_y * B_x * B_{x_i} * q_{y_i})
    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)B[i].bsums);
    const __m128i q8s =
        _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
    const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

    // get scales and broadcast to 512 bits
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m512i scales = _mm512_broadcast_i32x4(sc128);

    __m512i sum_i = _mm512_setzero_si512();

    const uint8_t* __restrict__ q4 = A[i].qs;
    const int8_t* __restrict__ q8 = B[i].qs;

#pragma unroll
    for (int j = 0; j < 256 / 128; ++j) {
      const __m512i scale_l = _mm512_shuffle_epi8(scales, get_scale_shuffle_avx512(2 * j));
      const __m512i scale_h = _mm512_shuffle_epi8(scales, get_scale_shuffle_avx512(2 * j + 1));

      const __m512i q4bits = _mm512_loadu_si512((const __m512i*)q4);
      q4 += 64;

      const __m512i q4l = _mm512_and_si512(q4bits, m4);
      __m512i q4h = _mm512_srli_epi16(q4bits, 4);
      q4h = _mm512_and_si512(q4h, m4);

      const __m512i q8l = _mm512_loadu_si512((const __m512i*)q8);
      q8 += 64;
      const __m512i q8h = _mm512_loadu_si512((const __m512i*)q8);
      q8 += 64;

      __m512i p16l = _mm512_maddubs_epi16(q4l, q8l);
      p16l = _mm512_madd_epi16(scale_l, p16l);

      __m512i p16h = _mm512_maddubs_epi16(q4h, q8h);
      p16h = _mm512_madd_epi16(scale_h, p16h);

      const __m512i sum_j = _mm512_add_epi32(p16l, p16h);

      sum_i = _mm512_add_epi32(sum_i, sum_j);
    }

    __m512 vd = _mm512_set1_ps(d);
    acc = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sum_i), acc);
  }

  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

  float acc_sum = _mm512_reduce_add_ps(acc) + _mm_cvtss_f32(acc_m);
  *C = acc_sum;
}

}  // namespace mllm::X86

#endif