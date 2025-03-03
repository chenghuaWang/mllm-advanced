/**
 * @file reduce.cu
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-03
 *
 * @copyright Copyright (c) 2025
 *
 */
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

}  // namespace mllm::cuda
