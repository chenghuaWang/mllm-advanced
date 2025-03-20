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
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val += __shfl_xor_sync(0xffffffff, val, lane_mask, width);
    return val;
  }
};

template<typename T>
struct WarpReduceMulOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val *= __shfl_xor_sync(0xffffffff, val, lane_mask, width);
    return val;
  }
};

template<typename T>
struct WarpReduceMinOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val = mllm_math::min(val, __shfl_xor_sync(0xffffffff, val, lane_mask, width));
    return val;
  }
};

template<typename T>
struct WarpReduceMaxOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val = mllm_math::max(val, __shfl_xor_sync(0xffffffff, val, lane_mask, width));
    return val;
  }
};

template<typename ReduceOp, typename T, int WARP_SIZE = 32>
__forceinline__ __device__ T warp_reduce(T val) {
  // loop from WARP_SIZE >> 1 to 1 instead of 1 to WARP_SIZE >> 1 for avoid bank conflict.
#pragma unroll
  for (int lane_mask = WARP_SIZE >> 1; lane_mask >= 1; lane_mask >>= 1) {
    val = ReduceOp::reduce(val, lane_mask, WARP_SIZE);
  }
  return val;
};

// ============================================================================================
// Block level reduce operation.
// ============================================================================================
// N_TILE actually is the threads num in a block.
template<int N_TILE = 256>
__device__ void block_all_reduce_sum_fp32(float* z, float* x, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * N_TILE + tid;

  // warp size is always 32.
  constexpr int NUM_WARPS = (N_TILE + 32 - 1) / 32;

  // shared memory to store the sum of each warp.
  __shared__ float smem_reduce[NUM_WARPS];

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  float sum = (idx < N) ? x[idx] : 0.f;
  sum = warp_reduce<WarpReduceAddOp<float>>(sum);
  if (lane_id == 0) smem_reduce[warp_id] = sum;

  // sync to make sure all warps have finished its reduce work.
  __syncthreads();

  // use the first warp to final reduce the sum of all warps.
  sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.f;
  if (warp_id == 0) sum = warp_reduce<WarpReduceAddOp<float>>(sum);
  if (tid == 0) atomicAdd(z, sum);
}

template<int N_TILE = 256>
__device__ void block_all_reduce_max_fp32(float* z, float* x, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * N_TILE + tid;

  // warp size is always 32.
  constexpr int NUM_WARPS = (N_TILE + 32 - 1) / 32;

  // shared memory to store the max of each warp.
  __shared__ float smem_reduce[NUM_WARPS];

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  float max_v = (idx < N) ? x[idx] : mllm_math::numeric_limits_min<float>();
  max_v = warp_reduce<WarpReduceMaxOp<float>>(max_v);
  if (lane_id == 0) smem_reduce[warp_id] = max_v;

  // sync to make sure all warps have finished its reduce work.
  __syncthreads();

  // use the first warp to final reduce the sum of all warps.
  max_v = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : mllm_math::numeric_limits_min<float>();
  if (warp_id == 0) max_v = warp_reduce<WarpReduceMaxOp<float>>(max_v);
  if (tid == 0) mllm_math::atomicMax(z, max_v);
}

template<int N_TILE = 256 / 8>
__device__ void block_all_reduce_sum_bf16x8_bf16(__nv_bfloat16* z, __nv_bfloat16* x, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * N_TILE + tid) * 8;
  constexpr int NUM_WARPS = (N_TILE + 32 - 1) / 32;

  __shared__ __nv_bfloat16 smem_reduce[NUM_WARPS];

  int warp = tid / 32;
  int lane = tid % 32;

  // load 128 bits
  __nv_bfloat16 pack[8];
  reinterpret_cast<float4*>(pack)[0] = reinterpret_cast<float4*>(x + idx)[0];

  const __nv_bfloat16 _0 = mllm_math::numeric_limits_pos_zero<__nv_bfloat16>();

  __nv_bfloat16 acc = _0;
#pragma unroll
  for (int i = 0; i < 8; ++i) { acc += (((idx + i) < N) ? pack[i] : _0); }

  acc = warp_reduce<WarpReduceAddOp<__nv_bfloat16>>(acc);
  if (lane == 0) smem_reduce[warp] = acc;
  __syncthreads();

  __nv_bfloat16 sum = (lane < NUM_WARPS) ? smem_reduce[lane] : _0;
  if (warp == 0) sum = warp_reduce<WarpReduceAddOp<__nv_bfloat16>>(sum);
  if (tid == 0) atomicAdd(z, __bfloat162float(sum));
}

}  // namespace mllm::cuda
