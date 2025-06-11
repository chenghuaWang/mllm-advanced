/**
 * @file fp32_s8s_pt.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>

namespace mllm::X86 {

/**
 * @brief Per Tensor quantization from float32 to int8. Using signed symmetry int8 quantization
 * method.
 *
 * NOTE: Only can be used offline. For preparing linear's weights, etc... The performance of this
 * quantization operator is not guaranteed.
 *
 * @param Z
 * @param scale
 * @param X
 * @param sequence
 * @param dim
 */
void fp32_s8s_pt_2d_offline(int8_t* __restrict__ Z, float* __restrict__ scale,
                            const float* __restrict__ X, int sequence, int dim);

}  // namespace mllm::X86
