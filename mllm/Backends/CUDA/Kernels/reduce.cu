/**
 * @file reduce.cu
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/Kernels/math_func.cuh"
#include "mllm/Backends/CUDA/Kernels/reduce.cuh"

namespace mllm::cuda {

template<int N_TILE>
__global__ void block_all_reduce_sum_fp32(float* z, float* x, int N) {
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

template<int N_TILE>
__global__ void block_all_reduce_sum_bf16x8_bf16(__nv_bfloat16* z, __nv_bfloat16* x, int N) {
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
