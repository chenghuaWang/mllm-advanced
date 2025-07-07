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
                                         int32_t kernel_size_h, int32_t kernel_size_w) {
  // Calculate output dimensions
  const int time_out = time - kernel_size_t + 1;
  const int h_out = h - kernel_size_h + 1;
  const int w_out = w - kernel_size_w + 1;
  const int kernel_vol = kernel_size_t * kernel_size_h * kernel_size_w;
  const int col_size = in_channels * kernel_vol;
  const int img_stride = time * h * w;  // Stride per image (all channels)
  const int chan_stride = h * w;        // Stride per channel (within one time slice)

  float* col_ptr = Z;

  for (int b = 0; b < batch; ++b) {
    const float* A_batch = A + b * in_channels * time * h * w;

    for (int t_out = 0; t_out < time_out; ++t_out) {
      for (int i_out = 0; i_out < h_out; ++i_out) {
        for (int j_out = 0; j_out < w_out; ++j_out) {
          // Process each kernel position
          for (int kt = 0; kt < kernel_size_t; ++kt) {
            const int t_in = t_out + kt;
            const float* A_time = A_batch + t_in * chan_stride;

            for (int kh = 0; kh < kernel_size_h; ++kh) {
              const int i_in = i_out + kh;
              const float* A_row = A_time + i_in * w;

              for (int kw = 0; kw < kernel_size_w; ++kw) {
                const int j_in = j_out + kw;
                const float* src_base = A_row + j_in;

                // Copy channels with NEON vectorization
                int c = 0;
                // Process 4 channels at a time
                for (; c <= in_channels - 4; c += 4) {
                  // Calculate channel offsets (each channel separated by chan_stride)
                  const float* src0 = src_base + c * img_stride;
                  const float* src1 = src0 + img_stride;
                  const float* src2 = src1 + img_stride;
                  const float* src3 = src2 + img_stride;

                  // Load single value from each channel
                  float32x4_t val = {*src0, *src1, *src2, *src3};
                  vst1q_f32(col_ptr, val);
                  col_ptr += 4;
                }
                // Process remaining channels (1-3)
                for (; c < in_channels; ++c) { *col_ptr++ = *(src_base + c * img_stride); }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace mllm::arm