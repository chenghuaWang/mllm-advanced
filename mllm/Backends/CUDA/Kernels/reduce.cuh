/**
 * @file reduce.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Backends/CUDA/Kernels/math_func.cuh"

namespace mllm::cuda {

// ============================================================================================
// Warp level reduce operation.
//
// The warp size is always set to 32. I use __shfl_xor_sync instead of __shfl_down_sync in warp
// level reduce.
// ============================================================================================
template<typename T>
struct WarpReduceAddOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask) {
    val += __shfl_xor_sync(0xffffffff, val, lane_mask, 32);
    return val;
  }
};

template<typename T>
struct WarpReduceMulOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask) {
    val *= __shfl_xor_sync(0xffffffff, val, lane_mask, 32);
    return val;
  }
};

template<typename T>
struct WarpReduceMinOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask) {
    val = mllm_math::min(val, __shfl_xor_sync(0xffffffff, val, lane_mask, 32));
    return val;
  }
};

template<typename T>
struct WarpReduceMaxOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask) {
    val = mllm_math::min(max, __shfl_xor_sync(0xffffffff, val, lane_mask, 32));
    return val;
  }
};

template<typename ReduceOp, typename T, int WARP_SIZE = 32>
__forceinline__ __device__ T warp_reduce(T val) {
#pragma unroll
  for (int lane_mask = WARP_SIZE >> 1; lane_mask >= 1; lane_mask >>= 1) {
    val = ReduceOp::reduce(val, lane_mask);
  }
  return val;
};

// ============================================================================================
// Block level reduce operation.
// ============================================================================================
template<int N_TILE = 256>
__global__ void block_all_reduce_sum_fp32(float* z, float* x, int N);

}  // namespace mllm::cuda
