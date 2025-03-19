/**
 * @file softmax.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cassert>
#include "mllm/Backends/CUDA/Kernels/gpu_info.cuh"
#include "mllm/Backends/CUDA/Kernels/math_func.cuh"
#include "mllm/Backends/CUDA/Kernels/reduce.cuh"
#include "mllm/Backends/CUDA/Kernels/softmax.cuh"

namespace mllm::cuda {

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
// instance
template __global__ void _warp_level_safe_softmax_fp32<32, 1, 32>(float* __restrict__ z,
                                                                  const float* __restrict__ x,
                                                                  int rows, int cols);
template __global__ void _warp_level_safe_softmax_fp32<16, 2, 32>(float* __restrict__ z,
                                                                  const float* __restrict__ x,
                                                                  int rows, int cols);
template __global__ void _warp_level_safe_softmax_fp32<8, 2, 32>(float* __restrict__ z,
                                                                 const float* __restrict__ x,
                                                                 int rows, int cols);
}  // namespace mllm::cuda