/**
 * @file conv3d.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <arm_neon.h>

namespace mllm::arm {

// padding is always 0
// dilation is always 1
void im2col_conv3d_p0_d1_activation_fp32(float* __restrict__ Z, const float* __restrict__ A,
                                         int32_t batch, int32_t in_channels, int32_t time,
                                         int32_t h, int32_t w, int32_t kernel_size_t,
                                         int32_t kernel_size_h, int32_t kernel_size_w,
                                         int32_t stride_size_t, int32_t stride_size_h,
                                         int32_t stride_size_w);

}  // namespace mllm::arm
