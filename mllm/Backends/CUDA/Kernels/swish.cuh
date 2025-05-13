/**
 * @file swish.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cute/tensor.hpp>
#include "cute/tensor_impl.hpp"
#include "mllm/Backends/CUDA/Kernels/math_func.cuh"

namespace mllm::cuda {

namespace {

__device__ __forceinline__ float4 v_swish_fp32(float4 v) {
  float4 ret;
  ret.x = v.x / (1.f + expf(-v.x));
  ret.y = v.y / (1.f + expf(-v.y));
  ret.z = v.z / (1.f + expf(-v.z));
  ret.w = v.w / (1.f + expf(-v.w));
  return ret;
}

}  // namespace

// n blocks and t threads in each block.
__global__ void swish_fp32(float* __restrict__ z, const float* __restrict__ x, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 rf_x, rf_y;
    MLLM_LDG128(&rf_x, x + idx);
    rf_y = v_swish_fp32(rf_x);
    MLLM_STG128(z + idx, &rf_y);
  }
}

template<int NUM_ELE_PER_THREAD = 4>
__global__ void swish_super_cute_fp32(float* __restrict__ z, const float* __restrict__ x, int N) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(N));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(N));

  Tensor tzr = local_tile(tz, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));

  // register file
  Tensor tzR = make_tensor_like(tzr);
  Tensor txR = make_tensor_like(txr);

  // LDG.128
  copy(txr, txR);

  auto tzR4 = recast<float4>(tzR);
  auto txR4 = recast<float4>(txR);

// compute
#pragma unroll
  for (int i = 0; i < size(tzR4); ++i) { tzR4[i] = v_swish_fp32(txR4[i]); }

  auto tzRx = recast<float>(tzR4);

  // STG.128
  copy(tzRx, tzr);
}
}  // namespace mllm::cuda