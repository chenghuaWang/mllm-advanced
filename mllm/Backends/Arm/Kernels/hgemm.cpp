/**
 * @file hgemm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else
#include <algorithm>
#include <arm_neon.h>

// Include micro-kernel variants
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

namespace mllm::arm {

namespace {

void hgemm_mk_kn_mn(const float16_t* __restrict lhs, const float16_t* __restrict rhs,
                    float16_t* __restrict dst, const size_t M, const size_t K, const size_t N,
                    const float16_t* __restrict bias) {
  /// Micro-kernel interface
  constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
      kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
      kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

  const size_t nr = ukernel.get_nr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();

  const size_t rhs_packed_size =
      kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
  const size_t rhs_packed_cols = nr + K * nr;
  const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float16_t));

  auto rhs_packed = new float16_t[rhs_packed_size];

  const size_t lhs_stride = K * sizeof(float16_t);
  const size_t rhs_stride = N * sizeof(float16_t);
  const size_t dst_stride_row = N * sizeof(float16_t);
  const size_t dst_stride_col = sizeof(float16_t);

  // pack once
  kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1, N, K, nr, kr, sr,  // Packing arguments
                                                    rhs_stride,           // RHS stride
                                                    rhs,                  // RHS
                                                    bias,                 // Bias
                                                    nullptr,              // Scale
                                                    rhs_packed,           // RHS packed
                                                    0, nullptr);

  const size_t m_step = ukernel.get_m_step();  // Scheduling along M
  const size_t n_step = ukernel.get_n_step();  // Scheduling along N

  for (size_t i_m_step = 0; i_m_step < M; i_m_step += m_step) {
    for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step) {
      // Support functions return offset in bytes
      const uint8_t* lhs_ptr =
          (const uint8_t*)lhs + (ukernel.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
      const uint8_t* rhs_ptr =
          (const uint8_t*)rhs_packed + (ukernel.get_rhs_packed_offset(i_n_step, K));
      uint8_t* dst_ptr =
          (uint8_t*)dst + (ukernel.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));

      const size_t actual_m = std::min(M - i_m_step, m_step);
      const size_t actual_n = std::min(N - i_n_step, n_step);

      ukernel.run_matmul(actual_m, actual_n, K,  // Dimensions
                         lhs_ptr,                // LHS
                         lhs_stride,             // LHS stride
                         rhs_ptr,                // RHS packed
                         dst_ptr,                // DST
                         dst_stride_row,         // DST stride (row)
                         dst_stride_col,         // DST stride (col)
                         -FLT_MAX, FLT_MAX       // Min and max for the clamp operation
      );
    }
  }

  delete[] rhs_packed;
}

}  // namespace

}  // namespace mllm::arm

#endif