/**
 * @file OpSelection.cu
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstdio>
#include "mllm/Backends/CUDA/Ops/OpSelection.hpp"
#include "mllm/Backends/CUDA/Kernels/swish.cuh"
#include "mllm/Backends/CUDA/Kernels/reduce.cuh"
#include "mllm/Backends/CUDA/Kernels/elewise.cuh"
#include "mllm/Backends/CUDA/Kernels/softmax.cuh"

namespace mllm::cuda {

void vector_add_bf16_v0_call(void* Z, void* const X, void* const Y, int size, float a, float b,
                             float c) {
  // calculate 8 elements in one block
  // 4 + 4 + 4 + 3 = 15 register is needed
  constexpr int block_size =
      1024;  // Max setting is MLLM_CUDA_PER_SM_32BIT_REGISTER_NUM / 15 = 4096

  // FIXME: consider sm count and warps use grid_size = max(1, min(x, hard_ware_limits))
  int grid_size = (size + block_size * 8 - 1) / (block_size * 8);

  vector_add_bf16_v0<8>
      <<<grid_size, block_size>>>((nv_bfloat16*)Z, (nv_bfloat16*)X, (nv_bfloat16*)Y, size, a, b, c);
  cudaError_t syncError = cudaGetLastError();
  if (syncError != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(syncError));
  }
}

void safe_softmax_fp32(void* __restrict__ Z, const void* __restrict__ X, int rows, int cols) {
  // CASE 1:
  // When cols < 1024. use one warp(32 threads) to calculate one/two row.
  if (cols <= 1024) {
    // FIXME: padding to multiple of warp size.

    // CASE 1.1:
    // 512 < cols <= 1024. One row per thread
    if (cols > 512) {
      dim3 block_dims(32, 8);              // 256 threads per block
      dim3 grid_dims((rows + 8 - 1) / 8);  // ceil_div(rows, 8)
      _warp_level_safe_softmax_fp32<32, 1, 32>
          <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
      return;
    }

    // CASE 1.2:
    // 256 < cols <= 512. Two row per threads.
    if (cols > 256) {
      dim3 block_dims(16, 8);              // 256 threads per block
      dim3 grid_dims((rows + 8 - 1) / 8);  // ceil_div(rows, 8)
      if (rows % 2 == 0) {
        _warp_level_safe_softmax_fp32<16, 2, 32>
            <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
      } else {
        _warp_level_safe_softmax_fp32<16, 1, 32>
            <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
      }
      return;
    }

    // CASE 1.3:
    // cols <= 256
    dim3 block_dims(8, 8);               // 64 threads per block
    dim3 grid_dims((rows + 8 - 1) / 8);  // ceil_div(rows, 8)
    if (rows % 2 == 0) {
      _warp_level_safe_softmax_fp32<8, 2, 32>
          <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
    } else {
      _warp_level_safe_softmax_fp32<8, 1, 32>
          <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
    }
    return;
  }

  // CASE 2:
  // cols > 1024
  dim3 block_dims(1024);  // 1024 threads per block
  dim3 grid_dims(32);
  _block_level_uncached_safe_softmax_fp32<1024>
      <<<grid_dims, block_dims, 0>>>((float*)Z, (float*)X, rows, cols);
}

void array_reduce_sum_fp32(void* __restrict__ Z, const void* __restrict__ X, int num) {
  // num <= 128, one block with 32 threads is enough
  if (num <= 128) {
    _1d_array_reduce_warp_level<WarpReduceAddOp<float>, float, 4>
        <<<1, 32>>>((float*)Z, (float*)X, num);
    return;
  }
  if (num <= 4096) {
    dim3 block_dims((num + 4 - 1) / 4);
    dim3 grid_dims(1);
#define DEF_OP_SELECT_CONDITION_WHEN(condition, sizes)                    \
  if ((condition)) {                                                      \
    _1d_array_reduce_block_level<WarpReduceAddOp<float>, float, 4, sizes> \
        <<<grid_dims, block_dims>>>((float*)Z, (float*)X, num);           \
    return;                                                               \
  }

    DEF_OP_SELECT_CONDITION_WHEN(block_dims.x <= 64, 64)
    DEF_OP_SELECT_CONDITION_WHEN(block_dims.x <= 128, 128)
    DEF_OP_SELECT_CONDITION_WHEN(block_dims.x <= 256, 256)
    DEF_OP_SELECT_CONDITION_WHEN(block_dims.x <= 512, 512)
    DEF_OP_SELECT_CONDITION_WHEN(block_dims.x <= 1024, 1024)

#undef DEF_OP_SELECT_CONDITION_WHEN
    return;
  }

  // num > 4096
  dim3 block_dims(1024);
  dim3 grid_dims((num + 4096 - 1) / 4096);  // FIXME: consider sm count
  _1d_array_reduce_grid_level<WarpReduceAddOp<float>, float, 4, 1024>
      <<<grid_dims, block_dims>>>((float*)Z, (float*)X, num);
  return;
}

void vector_swish_fp32_call(float* __restrict__ z, const float* __restrict__ x, int N) {
  constexpr int vector_size = 4;
  constexpr int threads_in_each_block = 128;
  dim3 block_dims(threads_in_each_block);
  dim3 grid_dims((N + vector_size * threads_in_each_block - 1)
                 / (vector_size * threads_in_each_block));
  swish_fp32<<<grid_dims, block_dims>>>(z, x, N);
}

void super_cute_swish_fp32_call(float* __restrict__ z, const float* __restrict__ x, int N) {
  constexpr int threads_in_each_block = 128;

  // CASE 1: N <= 128 * 8
  if (N <= 128 * 8) {
#define DEF_OP_SELECT_CONDITION_WHEN(condition, num_per_thread)                \
  if ((condition)) {                                                           \
    dim3 block_dims(threads_in_each_block);                                    \
    dim3 grid_dims((N + num_per_thread * threads_in_each_block - 1)            \
                   / (num_per_thread * threads_in_each_block));                \
    swish_super_cute_fp32<num_per_thread><<<grid_dims, block_dims>>>(z, x, N); \
    return;                                                                    \
  }

    // num_per_thread must be multiple of 4
    DEF_OP_SELECT_CONDITION_WHEN(N <= 128 * 4, 4);
    DEF_OP_SELECT_CONDITION_WHEN(N <= 128 * 8, 8);

#undef DEF_OP_SELECT_CONDITION_WHEN
    return;
  }

  // CASE 2: N > 128 * 8
  dim3 block_dims(threads_in_each_block);
  dim3 grid_dims((N + 128 * 4 * threads_in_each_block - 1) / (128 * 4 * threads_in_each_block));
  swish_super_cute_fp32<4><<<grid_dims, block_dims>>>(z, x, N);
}

}  // namespace mllm::cuda