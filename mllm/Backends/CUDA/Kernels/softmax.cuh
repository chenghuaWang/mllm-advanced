/**
 * @file softmax.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-02
 *
 * @copyright Copyright (c) 2025
 *
 * Some references:
 * 1. https://zhuanlan.zhihu.com/p/341059988 (oneflow's impl)
 * 2. https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
 *
 * This softmax impl is highly inspired by oneflow's blog.
 *
 */
#pragma once

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "mllm/Backends/CUDA/Kernels/gpu_info.cuh"
#include "mllm/Backends/CUDA/Kernels/math_func.cuh"
#include "mllm/Backends/CUDA/Kernels/reduce.cuh"

namespace mllm::cuda {

// impl block level reduce using cub in cccl
namespace {
template<int BLOCK_SIZE>
__inline__ __device__ float __cub_block_all_reduce_max(float val) {
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float result_broadcast;
  float result = BlockReduce(temp_storage).Reduce(val, cub::Max());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}

template<int BLOCK_SIZE>
__inline__ __device__ float __cub_block_all_reduce_sum(float val) {
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float result_broadcast;
  float result = BlockReduce(temp_storage).Sum(val);
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}
}  // namespace

// One warp to process one line, packed 4 float
// for cols <= 1024
template<int THREAD_GROUP_NUM, int ROWS_PER_THREAD, int COLS_PER_THREAD>
__global__ void _warp_level_safe_softmax_fp32(float* __restrict__ z, const float* __restrict__ x,
                                              int rows, int cols) {
  static_assert(COLS_PER_THREAD % 4 == 0);
  static_assert(THREAD_GROUP_NUM <= MLLM_CUDA_WARP_SIZE);
  static_assert(THREAD_GROUP_NUM == 32 || THREAD_GROUP_NUM == 16 || THREAD_GROUP_NUM == 8);
  static_assert(ROWS_PER_THREAD <= 2);

  assert(cols <= COLS_PER_THREAD * THREAD_GROUP_NUM);

  constexpr int num_packs = COLS_PER_THREAD / 4;

  float4 rX[ROWS_PER_THREAD][num_packs];

  const int this_thread_row = blockIdx.x * blockDim.y + threadIdx.y;
  const int grids_all_rows = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;

  for (int64_t row = this_thread_row * ROWS_PER_THREAD; row < rows;
       row += grids_all_rows * ROWS_PER_THREAD) {
    float local_max[ROWS_PER_THREAD];

#pragma unroll
    for (int row_id = 0; row_id < ROWS_PER_THREAD; ++row_id) {
      local_max[row_id] = mllm_math::numeric_limits_min<float>();
      float4* row_rX_ptr = rX[row_id];

// pass 1: max
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        // Note the memory access pattern below.
        //
        // Not use lane_id * COLS_PER_THREAD + pack_id * 4. This method is in-efficient due to
        // memory is not visited coalesced !!!
        const int col = (pack_id * THREAD_GROUP_NUM + lane_id) * 4;

        if (col < cols) {
          MLLM_LDG128(row_rX_ptr + pack_id, x + (row + row_id) * cols + col);
          float4 row_rX = *(row_rX_ptr + pack_id);
          local_max[row_id] = mllm_math::max(mllm_math::max(row_rX.x, row_rX.y),
                                             mllm_math::max(row_rX.z, row_rX.w));
        } else {
          row_rX_ptr[pack_id].x = mllm_math::numeric_limits_min<float>();
          row_rX_ptr[pack_id].y = mllm_math::numeric_limits_min<float>();
          row_rX_ptr[pack_id].z = mllm_math::numeric_limits_min<float>();
          row_rX_ptr[pack_id].w = mllm_math::numeric_limits_min<float>();
        }
      }
    }

    // pass 1: reduce local_max at warp level.
    float warp_max[ROWS_PER_THREAD];
#pragma unroll
    for (int row_id = 0; row_id < ROWS_PER_THREAD; ++row_id) {
      warp_max[row_id] =
          warp_reduce<WarpReduceMaxOp<float>, float, THREAD_GROUP_NUM>(local_max[row_id]);
    }

    // pass 2: sum all exp(x - warp_max)
    float local_sum[ROWS_PER_THREAD];
#pragma unroll
    for (int row_id = 0; row_id < ROWS_PER_THREAD; ++row_id) {
      local_sum[row_id] = mllm_math::numeric_limits_pos_zero<float>();
      float4* row_rX_ptr = rX[row_id];

#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        row_rX_ptr[pack_id].x = mllm_math::fast_exp(row_rX_ptr[pack_id].x - warp_max[row_id]);
        row_rX_ptr[pack_id].y = mllm_math::fast_exp(row_rX_ptr[pack_id].y - warp_max[row_id]);
        row_rX_ptr[pack_id].z = mllm_math::fast_exp(row_rX_ptr[pack_id].z - warp_max[row_id]);
        row_rX_ptr[pack_id].w = mllm_math::fast_exp(row_rX_ptr[pack_id].w - warp_max[row_id]);
        local_sum[row_id] += row_rX_ptr[pack_id].x + row_rX_ptr[pack_id].y + row_rX_ptr[pack_id].z
                             + row_rX_ptr[pack_id].w;
      }
    }

    // pass 2: reduce local_sum at warp level.
    float warp_sum[ROWS_PER_THREAD];
#pragma unroll
    for (int row_id = 0; row_id < ROWS_PER_THREAD; ++row_id) {
      warp_sum[row_id] =
          warp_reduce<WarpReduceAddOp<float>, float, THREAD_GROUP_NUM>(local_sum[row_id]);
    }

// pass 3: rescale and store.
#pragma unroll
    for (int row_id = 0; row_id < ROWS_PER_THREAD; ++row_id) {
      float4* row_rX_ptr = rX[row_id];

#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        float scale = 1.f / warp_sum[row_id];
        row_rX_ptr[pack_id].x *= scale;
        row_rX_ptr[pack_id].y *= scale;
        row_rX_ptr[pack_id].z *= scale;
        row_rX_ptr[pack_id].w *= scale;
      }

#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * THREAD_GROUP_NUM + lane_id) * 4;
        if (col < cols) { MLLM_STG128(z + (row + row_id) * cols + col, row_rX_ptr + pack_id); }
      }
    }
  }
}

// One block to process one line, packed 4 float
// For cols > 1024. This impl is uncached version, which not used shared memory. However, this
// method will reads `x` 3 times in each thread, which may lead to cache miss. Hence, the bigger the
// block size is, the better performance this kernel will give. In details, when the number of
// Blocks residing in the SM decreases, the amount of Cache space that each Block can exclusively
// occupy increases. If a large number of Blocks reside in the SM, the Cache will be frequently
// replaced → data may be squeezed out of the Cache, leading to the necessity of reloading data from
// global memory upon repeated access.
template<int BLOCK_SIZE, int PACK_SIZE = 4>
__global__ void _block_level_uncached_safe_softmax_fp32(float* __restrict__ z,
                                                        const float* __restrict__ x, int rows,
                                                        int cols) {
  static_assert(PACK_SIZE == 4);
  assert(cols % PACK_SIZE == 0);
  const int tid = threadIdx.x;

  const int num_packs = cols / PACK_SIZE;

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float local_max = mllm_math::numeric_limits_min<float>();

    // Pass 1: find max in this thread
    for (int pack_id = tid; pack_id < num_packs; pack_id += BLOCK_SIZE) {
      float4 tmp;
      MLLM_LDG128(&tmp, x + row * cols + pack_id * PACK_SIZE);

      local_max = mllm_math::max(mllm_math::max(tmp.x, tmp.y), mllm_math::max(tmp.z, tmp.w));
    }

    // Pass 1: reduce max values in this block(in one row)
    const float row_max = __cub_block_all_reduce_max<BLOCK_SIZE>(local_max);

    // Pass 2: sum all in this thread
    float local_sum = mllm_math::numeric_limits_pos_zero<float>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += BLOCK_SIZE) {
      float4 tmp;
      MLLM_LDG128(&tmp, x + row * cols + pack_id * PACK_SIZE);

      local_sum += mllm_math::fast_exp(tmp.x - row_max) + mllm_math::fast_exp(tmp.y - row_max)
                   + mllm_math::fast_exp(tmp.z - row_max) + mllm_math::fast_exp(tmp.w - row_max);
    }

    // Pass 2: reduce sum values in this block(in one row)
    float row_sum = __cub_block_all_reduce_sum<BLOCK_SIZE>(local_sum);
    row_sum = 1.f / row_sum;

    // Pass 3: rescale and store.
    for (int pack_id = tid; pack_id < num_packs; pack_id += BLOCK_SIZE) {
      float4 tmp;
      MLLM_LDG128(&tmp, x + row * cols + pack_id * PACK_SIZE);

      tmp.x = mllm_math::fast_exp(tmp.x - row_max) * row_sum;
      tmp.y = mllm_math::fast_exp(tmp.y - row_max) * row_sum;
      tmp.z = mllm_math::fast_exp(tmp.z - row_max) * row_sum;
      tmp.w = mllm_math::fast_exp(tmp.w - row_max) * row_sum;

      MLLM_STG128(z + row * cols + pack_id * PACK_SIZE, &tmp);
    }
  }
}

}  // namespace mllm::cuda