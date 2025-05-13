/**
 * @file ArmQuantizerHelper.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/ArmQuantizerHelper.hpp"

// for pack_kxn_fp16_w_bias
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

namespace mllm::arm {

void pack_kxn_fp16_w_bias_kleidiai(float16_t* __restrict__ packed_weight,
                                   const float16_t* __restrict__ weight,
                                   const float16_t* __restrict__ bias, int K, int N) {
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

  const size_t lhs_stride = K * sizeof(float16_t);
  const size_t rhs_stride = N * sizeof(float16_t);
  const size_t dst_stride_row = N * sizeof(float16_t);
  const size_t dst_stride_col = sizeof(float16_t);

  kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1, N, K, nr, kr, sr,  // Packing arguments
                                                    rhs_stride,           // RHS stride
                                                    weight,               // RHS
                                                    bias,                 // Bias
                                                    nullptr,              // Scale
                                                    packed_weight,        // RHS packed
                                                    0, nullptr);
}

}  // namespace mllm::arm
