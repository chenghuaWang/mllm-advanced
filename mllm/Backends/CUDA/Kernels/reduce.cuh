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

#include "mllm/Backends/CUDA/Kernels/gpu_info.cuh"
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

  static __forceinline__ __device__ T default_v() {
    return mllm_math::numeric_limits_pos_zero<T>();
  }

  static __forceinline__ __device__ T reduce_two(T a, T b) { return a + b; }
};

template<typename T>
struct WarpReduceMulOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val *= __shfl_xor_sync(0xffffffff, val, lane_mask, width);
    return val;
  }

  static __forceinline__ __device__ T default_v() { return 1; }

  static __forceinline__ __device__ T reduce_two(T a, T b) { return a * b; }
};

template<typename T>
struct WarpReduceMinOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val = mllm_math::min(val, __shfl_xor_sync(0xffffffff, val, lane_mask, width));
    return val;
  }

  static __forceinline__ __device__ T default_v() { return mllm_math::numeric_limits_max<T>(); }

  static __forceinline__ __device__ T reduce_two(T a, T b) { return mllm_math::min(a, b); }
};

template<typename T>
struct WarpReduceMaxOp {
  static __forceinline__ __device__ T reduce(T val, int lane_mask, int width) {
    val = mllm_math::max(val, __shfl_xor_sync(0xffffffff, val, lane_mask, width));
    return val;
  }

  static __forceinline__ __device__ T default_v() { return mllm_math::numeric_limits_min<T>(); }

  static __forceinline__ __device__ T reduce_two(T a, T b) { return mllm_math::max(a, b); }
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

// Reduce all `value` gave in this thread from one block. This function cannot handle multi block
// reduction. Which means this function do not need atomicAdd or atomicMax.
template<typename T, typename ReduceOp, int BLOCK_DIM_X, bool PADDING = true, int BLOCK_DIM_Y = 1,
         int BLOCK_DIM_Z = 1>
__device__ T block_reduce(T v, int num = -1) {
  constexpr int threads_num = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  constexpr int warps = (threads_num + MLLM_CUDA_WARP_SIZE - 1) / MLLM_CUDA_WARP_SIZE;

  // flatten threads to 1D
  int tid = threadIdx.x + threadIdx.y * BLOCK_DIM_X + threadIdx.z * BLOCK_DIM_X * BLOCK_DIM_Y;
  int warp_id = tid / MLLM_CUDA_WARP_SIZE;
  int lane_id = tid % MLLM_CUDA_WARP_SIZE;

  // alloc shared memory for communication
  __shared__ T s_reduced_in_each_warp[warps];

  // reduce in warp and store to block shared mem.
  if constexpr (PADDING) {
    T r_v = warp_reduce<ReduceOp, T, MLLM_CUDA_WARP_SIZE>(v);
    if (lane_id == 0) { s_reduced_in_each_warp[warp_id] = r_v; }
  } else {
    T _r_v = tid < num ? v : ReduceOp::default_v();
    T r_v = warp_reduce<ReduceOp, T, MLLM_CUDA_WARP_SIZE>(_r_v);
    if (lane_id == 0) { s_reduced_in_each_warp[warp_id] = r_v; }
  }

  // wait for s_reduced_in_each_warp filled.
  __syncthreads();

  __shared__ T result_global;

  // reduce in this block. the max threads in a block is 1024. which means warps<=32.
  if (warp_id == 0) {
    T tmp = lane_id < warps ? s_reduced_in_each_warp[lane_id] : ReduceOp::default_v();
    T r_r_v = warp_reduce<ReduceOp, T, MLLM_CUDA_WARP_SIZE>(tmp);
    if (lane_id == 0) result_global = r_r_v;
  }

  // sync for values store into result_global
  __syncthreads();
  return result_global;
}

// ============================================================================================
// Reduce operation for an array.
// ============================================================================================
template<typename ReduceOp, typename T, int VEC_SIZE>
struct _1DArrayReduceImpl {
  static __device__ void _warp_level(T* z, T* x, int num) {}

  template<int BLOCK_SIZE>
  static __device__ void _block_level(T* z, T* x, int num) {}

  template<int BLOCK_SIZE>
  static __device__ void _grid_level(T* z, T* x, int num) {}
};

template<typename ReduceOp>
struct _1DArrayReduceImpl<ReduceOp, float, 4> {
  static __device__ void _warp_level(float* z, float* x, int num) {
    int tid = threadIdx.x;
    int index = tid * 4;
    float4 tmp;

    if (index + 3 < num) {
      MLLM_LDG128(&tmp, x + index);
    } else if (index < num) {
      tmp.x = (index + 0 < num) ? x[index + 0] : ReduceOp::default_v();
      tmp.y = (index + 1 < num) ? x[index + 1] : ReduceOp::default_v();
      tmp.z = (index + 2 < num) ? x[index + 2] : ReduceOp::default_v();
      tmp.w = (index + 3 < num) ? x[index + 3] : ReduceOp::default_v();
    } else {
      tmp.x = tmp.y = tmp.z = tmp.w = ReduceOp::default_v();
    }

    float v = ReduceOp::reduce_two(ReduceOp::reduce_two(tmp.x, tmp.y),
                                   ReduceOp::reduce_two(tmp.z, tmp.w));

    float r_v = warp_reduce<ReduceOp, float, MLLM_CUDA_WARP_SIZE>(v);
    if (tid == 0) *z = r_v;
  }

  template<int BLOCK_SIZE>
  static __device__ void _block_level(float* z, float* x, int num) {
    // TODO
  }

  template<int BLOCK_SIZE>
  static __device__ void _grid_level(float* z, float* x, int num) {
    // TODO
  }
};

// Use one block with < 32 threads to reduce an array < 128 elements. Which means we can use warp
// level reduce.
//
// launch this kernel:
// _1d_array_reduce_warp_level<ReduceOp, T, VEC_SIZE><<<1, 32>>>(z, x, num);
template<typename ReduceOp, typename T, int VEC_SIZE>
__global__ void _1d_array_reduce_warp_level(T* z, T* x, int num) {
  _1DArrayReduceImpl<ReduceOp, T, VEC_SIZE>::_warp_level(z, x, num);
}

// Use one block with < 1024 threads to reduce an array < 4096 elements. Which means we can use
// block level reduce
//
// launch this kernel:
// _1d_array_reduce_block_level<ReduceOp, T, VEC_SIZE, 1024><<<1, 1024>>>(z, x, num);
// 32 < blockDim.x <= 1024
template<typename ReduceOp, typename T, int VEC_SIZE, int BLOCK_SIZE>
__global__ void _1d_array_reduce_block_level(T* z, T* x, int num) {
  _1DArrayReduceImpl<ReduceOp, T, VEC_SIZE>::_block_level<BLOCK_SIZE>(z, x, num);
}

// Use MULTI-block with < 1024 threads to reduce an array > 4096.
//
// launch this kernel:
// _1d_array_reduce_grid_level<ReduceOp, T, VEC_SIZE, 256><<<min(num/(256*4), sm), 256>>>(z, x,
// num);
//
// There has communication between blocks. And ATOMIC primitives is used
template<typename ReduceOp, typename T, int VEC_SIZE, int BLOCK_SIZE>
__global__ void _1d_array_reduce_grid_level(T* z, T* x, int num) {
  _1DArrayReduceImpl<ReduceOp, T, VEC_SIZE>::_grid_level<BLOCK_SIZE>(z, x, num);
}

}  // namespace mllm::cuda
