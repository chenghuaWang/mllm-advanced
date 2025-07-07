/**
 * @file conv3d.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Kernels/conv3d.hpp"

namespace mllm::arm {

void im2col_conv3d_p0_d1_activation_fp32(float* __restrict__ Z, const float* __restrict__ A,
                                         int32_t batch, int32_t in_channels, int32_t time,
                                         int32_t h, int32_t w, int32_t kernel_size_t,
                                         int32_t kernel_size_h, int32_t kernel_size_w,
                                         int32_t stride_size_t, int32_t stride_size_h,
                                         int32_t stride_size_w) {
  const int32_t out_time = (time - kernel_size_t) / stride_size_t + 1;
  const int32_t out_h = (h - kernel_size_h) / stride_size_h + 1;
  const int32_t out_w = (w - kernel_size_w) / stride_size_w + 1;

  const int32_t k_vol = kernel_size_t * kernel_size_h * kernel_size_w;
  const int32_t col_stride = in_channels * k_vol;
  const int32_t A_batch_stride = in_channels * time * h * w;
  const int32_t A_channel_stride = time * h * w;
  const int32_t A_time_stride = h * w;
  const int32_t A_h_stride = w;

  const int32_t neon_vectors = w / 4;
  const int32_t residual = w % 4;

  for (int32_t b = 0; b < batch; ++b) {
    const float* A_batch = A + b * A_batch_stride;

    for (int32_t ot = 0; ot < out_time; ++ot) {
      const int32_t t_start = ot * stride_size_t;

      for (int32_t oh = 0; oh < out_h; ++oh) {
        const int32_t h_start = oh * stride_size_h;

        for (int32_t ow = 0; ow < out_w; ++ow) {
          const int32_t w_start = ow * stride_size_w;
          const int32_t col_index = (b * out_time * out_h + ot * out_h + oh) * out_w + ow;
          float* Z_col = Z + col_index * col_stride;

          for (int32_t c = 0; c < in_channels; ++c) {
            const float* A_channel = A_batch + c * A_channel_stride;

            for (int32_t kt = 0; kt < kernel_size_t; ++kt) {
              const int32_t t = t_start + kt;
              const float* A_time = A_channel + t * A_time_stride;

              for (int32_t kh = 0; kh < kernel_size_h; ++kh) {
                const int32_t y = h_start + kh;
                const float* A_row = A_time + y * A_h_stride + w_start;

                const int32_t z_idx =
                    c * k_vol + kt * (kernel_size_h * kernel_size_w) + kh * kernel_size_w;
                float* Z_dest = Z_col + z_idx;

                int32_t kw = 0;
                for (; kw < neon_vectors * 4; kw += 4) {
                  float32x4_t v = vld1q_f32(A_row + kw);
                  vst1q_f32(Z_dest + kw, v);
                }

                for (; kw < kernel_size_w; ++kw) { Z_dest[kw] = A_row[kw]; }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace mllm::arm