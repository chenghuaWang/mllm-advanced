/**
 * @file fp32_s8s_pto_key.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-26
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Utils/Common.hpp"
#include "mllm/Utils/ThreadPool.hpp"
#include "mllm/Backends/Arm/Kernels/math.hpp"
#include "mllm/Backends/Arm/Kernels/Quantize/fp32_s8s_pto_key.hpp"

namespace mllm::arm {

void fp32_s8s_pto_key_bshd(int8_t* __restrict__ Z, float* __restrict__ scale,
                           const float* __restrict__ X, int B, int S, int H, int D, bool clamp,
                           float clamp_min, float clamp_max) {
  if (clamp) { NYI("clamp is not supported yet"); }

  for (int b = 0; b < B; ++b) {
    MLLM_PARALLEL_FOR(token_id, 0, S) {
      auto this_token_ptr = X + b * S * H * D + token_id * H * D;
      auto this_token_dst_ptr = Z + b * S * H * D + token_id * H * D;

      // 0. Get mean from X[b, token_id, :, :], And do K = K - mean(K)
      auto this_token_mean = vsum_reduce_fp32(this_token_ptr, H * D) / static_cast<float>(H * D);

      // 1. Find min and max from X[b, token_id, :, :]
      float this_token_max = -FLT_MAX;
      float this_token_min = FLT_MAX;
      int i = 0;
      const int N = H * D;

      {
        float32x4_t max_v = vdupq_n_f32(-FLT_MAX);
        float32x4_t min_v = vdupq_n_f32(FLT_MAX);
        const float32x4_t mean_v = vdupq_n_f32(this_token_mean);

        for (; i <= N - 16; i += 16) {
          float32x4_t x0 = vsubq_f32(vld1q_f32(this_token_ptr + i), mean_v);
          float32x4_t x1 = vsubq_f32(vld1q_f32(this_token_ptr + i + 4), mean_v);
          float32x4_t x2 = vsubq_f32(vld1q_f32(this_token_ptr + i + 8), mean_v);
          float32x4_t x3 = vsubq_f32(vld1q_f32(this_token_ptr + i + 12), mean_v);

          max_v = vmaxq_f32(max_v, x0);
          max_v = vmaxq_f32(max_v, x1);
          max_v = vmaxq_f32(max_v, x2);
          max_v = vmaxq_f32(max_v, x3);

          min_v = vminq_f32(min_v, x0);
          min_v = vminq_f32(min_v, x1);
          min_v = vminq_f32(min_v, x2);
          min_v = vminq_f32(min_v, x3);
        }

        for (; i <= N - 4; i += 4) {
          float32x4_t x = vsubq_f32(vld1q_f32(this_token_ptr + i), mean_v);
          max_v = vmaxq_f32(max_v, x);
          min_v = vminq_f32(min_v, x);
        }

        this_token_max = vmaxvq_f32(max_v);
        this_token_min = vminvq_f32(min_v);

        for (; i < N; ++i) {
          float val = this_token_ptr[i] - this_token_mean;
          this_token_max = fmax(this_token_max, val);
          this_token_min = fmin(this_token_min, val);
        }
      }

      // 2. Calculate scale = (max - min) / (2 ** 7 - 1)
      float this_token_scale = (this_token_max - this_token_min) / 127.0f;
      scale[token_id] = this_token_scale;
      float rescale_factor = 1.0f / this_token_scale;
      const float32x4_t rescale_v = vdupq_n_f32(rescale_factor);
      const float32x4_t mean_v_neon = vdupq_n_f32(this_token_mean);

      i = 0;
      for (; i <= N - 16; i += 16) {
        // Load and subtract mean
        float32x4_t x0 = vsubq_f32(vld1q_f32(this_token_ptr + i), mean_v_neon);
        float32x4_t x1 = vsubq_f32(vld1q_f32(this_token_ptr + i + 4), mean_v_neon);
        float32x4_t x2 = vsubq_f32(vld1q_f32(this_token_ptr + i + 8), mean_v_neon);
        float32x4_t x3 = vsubq_f32(vld1q_f32(this_token_ptr + i + 12), mean_v_neon);

        // Rescale
        x0 = vmulq_f32(x0, rescale_v);
        x1 = vmulq_f32(x1, rescale_v);
        x2 = vmulq_f32(x2, rescale_v);
        x3 = vmulq_f32(x3, rescale_v);

        // Convert to integers
        int32x4_t s0 = vcvtq_s32_f32(x0);
        int32x4_t s1 = vcvtq_s32_f32(x1);
        int32x4_t s2 = vcvtq_s32_f32(x2);
        int32x4_t s3 = vcvtq_s32_f32(x3);

        // Narrow and pack
        int16x4_t s0_16 = vqmovn_s32(s0);
        int16x4_t s1_16 = vqmovn_s32(s1);
        int16x4_t s2_16 = vqmovn_s32(s2);
        int16x4_t s3_16 = vqmovn_s32(s3);

        int16x8_t low = vcombine_s16(s0_16, s1_16);
        int16x8_t high = vcombine_s16(s2_16, s3_16);
        int8x8_t low8 = vqmovn_s16(low);
        int8x8_t high8 = vqmovn_s16(high);

        vst1q_s8(this_token_dst_ptr + i, vcombine_s8(low8, high8));
      }

      for (; i <= N - 8; i += 8) {
        float32x4_t x0 = vsubq_f32(vld1q_f32(this_token_ptr + i), mean_v_neon);
        float32x4_t x1 = vsubq_f32(vld1q_f32(this_token_ptr + i + 4), mean_v_neon);
        x0 = vmulq_f32(x0, rescale_v);
        x1 = vmulq_f32(x1, rescale_v);

        int32x4_t s0 = vcvtq_s32_f32(x0);
        int32x4_t s1 = vcvtq_s32_f32(x1);

        int16x4_t s0_16 = vqmovn_s32(s0);
        int16x4_t s1_16 = vqmovn_s32(s1);
        int8x8_t res = vqmovn_s16(vcombine_s16(s0_16, s1_16));

        vst1_s8(this_token_dst_ptr + i, res);
      }

      for (; i <= N - 4; i += 4) {
        float32x4_t x = vsubq_f32(vld1q_f32(this_token_ptr + i), mean_v_neon);
        x = vmulq_f32(x, rescale_v);
        int32x4_t s = vcvtq_s32_f32(x);
        int32_t s32[4];

        vst1q_s32(s32, s);

        for (int j = 0; j < 4; ++j) {
          s32[j] = std::max(-128, std::min(127, s32[j]));
          this_token_dst_ptr[i + j] = static_cast<int8_t>(s32[j]);
        }
      }

      for (; i < N; ++i) {
        float val = (this_token_ptr[i] - this_token_mean) * rescale_factor;

        int32_t q = static_cast<int32_t>(val);
        q = std::max(-128, std::min(127, q));
        this_token_dst_ptr[i] = static_cast<int8_t>(q);
      }
    }
    MLLM_PARALLEL_FOR_END;
  }
}

}  // namespace mllm::arm
