/**
 * @file kai_linear.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-05-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <algorithm>
#include "mllm/Utils/ThreadPool.hpp"
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

// for pack_kxn_fp16_w_bias
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

// for f32_qai8dxp_qsi4c32
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

namespace mllm::arm {

kai_matmul_clamp_f16_f16_f16p_ukernel KaiLinear_fp16_fp16_fp16p_mxk_kxn::ukernel_ = {
    .get_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_lhs_packed_offset =
        kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_rhs_packed_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

size_t KaiLinear_fp16_fp16_fp16p_mxk_kxn::pack_rhs_size(int K, int N) {
  return kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
}

void KaiLinear_fp16_fp16_fp16p_mxk_kxn::pack_rhs_offline(float16_t* __restrict__ rhs_packed,
                                                         const float16_t* __restrict__ rhs,
                                                         const float16_t* bias, int K, int N) {
  bool has_bias = bias != nullptr;
  float16_t* fake_bias = nullptr;

  if (!has_bias) {
    fake_bias = new float16_t[N];
    for (int i = 0; i < N; ++i) fake_bias[i] = 0;
  }

  const size_t nr = ukernel_.get_nr();
  const size_t kr = ukernel_.get_kr();
  const size_t sr = ukernel_.get_sr();

  const size_t rhs_stride = N * sizeof(float16_t);

  kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1, N, K, nr, kr, sr,  // Packing arguments
                                                    rhs_stride,           // RHS stride
                                                    rhs,                  // RHS
                                                    has_bias ? bias : fake_bias,  // Bias
                                                    nullptr,                      // Scale
                                                    rhs_packed,                   // RHS packed
                                                    0, nullptr);
  if (!has_bias) { delete[] fake_bias; }
}

void KaiLinear_fp16_fp16_fp16p_mxk_kxn::matmul(float16_t* __restrict__ dst,
                                               const float16_t* __restrict__ lhs,
                                               const float16_t* __restrict__ rhs, int M, int K,
                                               int N) {
  const int lhs_stride = K * sizeof(float16_t);
  const int dst_stride_row = N * sizeof(float16_t);
  const int dst_stride_col = sizeof(float16_t);

  const int m_step = ukernel_.get_m_step();  // Scheduling along M
  const int n_step = ukernel_.get_n_step();  // Scheduling along N

  MLLM_PARALLEL_FOR_CHUNK(i_m_step, 0, M, m_step) {
    for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
      // Support functions return offset in bytes
      const uint8_t* lhs_ptr =
          (const uint8_t*)lhs + (ukernel_.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
      const uint8_t* rhs_ptr = (const uint8_t*)rhs + (ukernel_.get_rhs_packed_offset(i_n_step, K));
      uint8_t* dst_ptr =
          (uint8_t*)dst + (ukernel_.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));

      const int actual_m = std::min(M - i_m_step, m_step);
      const int actual_n = std::min(N - i_n_step, n_step);

      ukernel_.run_matmul(actual_m, actual_n, K,  // Dimensions
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
  MLLM_PARALLEL_FOR_END;
}

std::unordered_map<KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles,
                   kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>
    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::ukernels_ = {
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_size =
              kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod

         }},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_size =
              kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_size =
              kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod}}};

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::quant_nxk_qs4c32_f32(size_t n, size_t k, size_t bl,
                                                                  const float* rhs_f32,
                                                                  uint8_t* rhs_qs4c32,
                                                                  uint16_t* rhs_scales_bf16) {
  constexpr int INT4_MIN = -8;
  constexpr int INT4_MAX = 7;

  const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
  const size_t rhs_qs4c32_stride = get_rhs_native_stride(k);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4c32, 0, n * rhs_qs4c32_stride);

  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float* src_ptr = rhs_f32 + row_idx * k;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;
      float max = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const size_t k_idx = block_idx * bl + b;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
          max = src0_0;
        }
      }

      const float scale = max / -8.0;
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale in the dedicated buffer
      *rhs_scales_bf16 = kai_cast_bf16_f32(scale);

      rhs_scales_bf16 += 1;

      for (size_t i = 0; i < bl; ++i) {
        const size_t k_idx = block_idx * bl + i;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];

        // Scale the values
        int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

        // Maximum/minimum int4 values
        v0_s32 = std::max(v0_s32, INT4_MIN);
        v0_s32 = std::min(v0_s32, INT4_MAX);

        const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

        const size_t dst_addr = (k_idx / 2) + row_idx * rhs_qs4c32_stride;
        uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

        if ((k_idx % 2) == 0) {
          rhs_v0 = v0_u8;
        } else {
          rhs_v0 |= (v0_u8 << 4);
        }

        rhs_qs4c32[dst_addr] = rhs_v0;
      }
    }
  }
}

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::workspace_size(
    int M, int K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t mr = ukernels_[tile_cfg].get_mr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
}

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::quant_pack_rhs_size(
    int N, int K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, 32,
                                                                   kai_dt_bf16);
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::quant_pack_rhs_offline(
    uint8_t* __restrict__ packed_weight, const float* __restrict__ rhs,
    const float* __restrict__ bias, int N, int K,
    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  // meta info
  const size_t rhs_native_size_f32 = N * K * sizeof(float);
  const size_t rhs_native_size_qs4c32 = N * get_rhs_native_stride(K);
  const size_t rhs_scales_size_bf16 = N * get_rhs_scale_stride(K, 32);

  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  uint8_t* rhs_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  uint8_t* rhs_scales_bf16 = new uint8_t[rhs_scales_size_bf16];

  // quant
  quant_nxk_qs4c32_f32(N, K, 32, rhs, rhs_qs4c32, (uint16_t*)rhs_scales_bf16);

  // pack
  kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;
  params.scale_dt = kai_dt_bf16;
  kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1, N, K,                       // Dimensions
                                            nr, kr, sr,                    // Packing arguments
                                            32,                            // Block length
                                            (const uint8_t*)(rhs_qs4c32),  // RHS
                                            get_rhs_native_stride(K),      // RHS stride
                                            bias,                          // Bias
                                            rhs_scales_bf16,               // Scale
                                            get_rhs_scale_stride(K, 32),   // Scale stride
                                            packed_weight,                 // RHS packed
                                            0, &params);

  delete[] rhs_qs4c32;
  delete[] rhs_scales_bf16;
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::matmul(
    float* __restrict__ dst, const float* __restrict__ lhs_fp32, const uint8_t* packed_weight_bias,
    void* workspace, int M, int K, int N, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  const size_t mr = ukernels_[tile_cfg].get_mr();

  kai_run_lhs_quant_pack_qai8dxp_f32(M, K,                    // Dimensions
                                     mr, kr, sr, 0,           // Packing arguments
                                     (const float*)lhs_fp32,  // LHS
                                     K * sizeof(float),       // LHS stride
                                     workspace);              // LHS packed

  // matmul
  {
    const size_t dst_stride = N * sizeof(float);
    const int m_step = ukernels_[tile_cfg].get_m_step();  // Scheduling along M
    const int n_step = ukernels_[tile_cfg].get_n_step();  // Scheduling along N

    MLLM_PARALLEL_FOR_CHUNK(i_m_step, 0, M, m_step) {
      for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
        // Support functions return offset in bytes
        const void* lhs_ptr =
            (const void*)((const char*)workspace
                          + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step, K)));
        const void* rhs_ptr =
            (const void*)((const char*)packed_weight_bias
                          + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
        float* dst_ptr =
            (float*)((uint8_t*)dst
                     + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

        const int actual_m = std::min(M - i_m_step, m_step);
        const int actual_n = std::min(N - i_n_step, n_step);

        ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
                                       32,                     // Block length
                                       lhs_ptr,                // LHS packed
                                       rhs_ptr,                // RHS packed
                                       dst_ptr,                // DST
                                       dst_stride,             // DST stride (row)
                                       sizeof(float),          // DST stride (col)
                                       -FLT_MAX, FLT_MAX);
      }
    }
    MLLM_PARALLEL_FOR_END;
  }
}

std::unordered_map<KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles,
                   kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>
    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::ukernels_ = {
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_size =
              kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod

         }},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_size =
              kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_size =
              kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_lhs_packed_offset =
              kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_rhs_packed_offset =
              kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_offset =
              kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod}}};

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::workspace_size(
    int M, int K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  const size_t mr = ukernels_[tile_cfg].get_mr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
}

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::quant_pack_rhs_size(
    int K, int N, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, 32,
                                                                   kai_dt_bf16);
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::quant_pack_rhs_offline(
    uint8_t* __restrict__ packed_weight, const float* __restrict__ rhs,
    const float* __restrict__ bias, int K, int N,
    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  // meta info
  const size_t rhs_native_size_f32 = N * K * sizeof(float);
  const size_t rhs_native_size_qs4c32 = N * get_rhs_native_stride(K);
  const size_t rhs_scales_size_bf16 = N * get_rhs_scale_stride(K, 32);

  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  uint8_t* rhs_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  uint8_t* rhs_scales_bf16 = new uint8_t[rhs_scales_size_bf16];

  // quant
  quant_kxn_qs4c32_f32(N, K, 32, rhs, rhs_qs4c32, (uint16_t*)rhs_scales_bf16);

  // pack
  kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;
  params.scale_dt = kai_dt_bf16;
  kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(1, N, K,                       // Dimensions
                                            nr, kr, sr,                    // Packing arguments
                                            32,                            // Block length
                                            (const uint8_t*)(rhs_qs4c32),  // RHS
                                            get_rhs_native_stride(K),      // RHS stride
                                            bias,                          // Bias
                                            rhs_scales_bf16,               // Scale
                                            get_rhs_scale_stride(K, 32),   // Scale stride
                                            packed_weight,                 // RHS packed
                                            0, &params);

  delete[] rhs_qs4c32;
  delete[] rhs_scales_bf16;
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::matmul(
    float* __restrict__ dst, const float* __restrict__ lhs_fp32, const uint8_t* packed_weight_bias,
    void* workspace, int M, int K, int N, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  const size_t mr = ukernels_[tile_cfg].get_mr();

  kai_run_lhs_quant_pack_qai8dxp_f32(M, K,                    // Dimensions
                                     mr, kr, sr, 0,           // Packing arguments
                                     (const float*)lhs_fp32,  // LHS
                                     K * sizeof(float),       // LHS stride
                                     workspace);              // LHS packed

  // matmul
  {
    const size_t dst_stride = N * sizeof(float);
    const int m_step = ukernels_[tile_cfg].get_m_step();  // Scheduling along M
    const int n_step = ukernels_[tile_cfg].get_n_step();  // Scheduling along N

    MLLM_PARALLEL_FOR_CHUNK(i_m_step, 0, M, m_step) {
      for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
        // Support functions return offset in bytes
        const void* lhs_ptr =
            (const void*)((const char*)workspace
                          + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step, K)));
        const void* rhs_ptr =
            (const void*)((const char*)packed_weight_bias
                          + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
        float* dst_ptr =
            (float*)((uint8_t*)dst
                     + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

        const int actual_m = std::min(M - i_m_step, m_step);
        const int actual_n = std::min(N - i_n_step, n_step);

        ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
                                       32,                     // Block length
                                       lhs_ptr,                // LHS packed
                                       rhs_ptr,                // RHS packed
                                       dst_ptr,                // DST
                                       dst_stride,             // DST stride (row)
                                       sizeof(float),          // DST stride (col)
                                       -FLT_MAX, FLT_MAX);
      }
    }
    MLLM_PARALLEL_FOR_END;
  }
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::quant_kxn_qs4c32_f32(size_t n, size_t k, size_t bl,
                                                                  const float* rhs_f32,
                                                                  uint8_t* rhs_qs4c32,
                                                                  uint16_t* rhs_scales_bf16) {
  constexpr int INT4_MIN = -8;
  constexpr int INT4_MAX = 7;

  const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
  const size_t rhs_qs4c32_stride = get_rhs_native_stride(n);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4c32, 0, k * rhs_qs4c32_stride);

  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float* src_ptr = rhs_f32 + row_idx * k;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;
      float max = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const size_t k_idx = block_idx * bl + b;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
          max = src0_0;
        }
      }

      const float scale = max / -8.0;
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale in the dedicated buffer
      *rhs_scales_bf16 = kai_cast_bf16_f32(scale);

      rhs_scales_bf16 += 1;

      for (size_t i = 0; i < bl; ++i) {
        const size_t k_idx = block_idx * bl + i;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];

        // Scale the values
        int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

        // Maximum/minimum int4 values
        v0_s32 = std::max(v0_s32, INT4_MIN);
        v0_s32 = std::min(v0_s32, INT4_MAX);

        const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

        const size_t dst_addr = (row_idx / 2) + k_idx * rhs_qs4c32_stride;
        uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

        if ((row_idx % 2) == 0) {
          rhs_v0 = v0_u8;
        } else {
          rhs_v0 |= (v0_u8 << 4);
        }

        rhs_qs4c32[dst_addr] = rhs_v0;
      }
    }
  }
}

std::unordered_map<KaiLinear_f16_qsi8d32p_qai4c32p_mxk_kxn::Tiles,
                   kai_matmul_clamp_f16_qsi8d32p_qai4c32p_ukernel>
    KaiLinear_f16_qsi8d32p_qai4c32p_mxk_kxn::ukernels_ = {
        // TODO
};

}  // namespace mllm::arm