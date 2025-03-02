/**
 * @file elewise.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mllm::cuda {

template<int NUM_ELE_PER_THREAD = 4>
__global__ void vector_add_f32_v0(float* z, const float* x, const float* y, int num, const float a,
                                  const float b, const float c);

// z = a * x + b * y + c
template<int NUM_ELE_PER_THREAD = 8>
__global__ void vector_add_bf16_v0(nv_bfloat16* z, const nv_bfloat16* x, const nv_bfloat16* y,
                                   int num, const nv_bfloat16 a, const nv_bfloat16 b,
                                   const nv_bfloat16 c);

// z = x - y
template<int NUM_ELE_PER_THREAD = 8>
__global__ void vector_sub_bf16_v0(nv_bfloat16* z, const nv_bfloat16* x, const nv_bfloat16* y,
                                   int num);

}  // namespace mllm::cuda
