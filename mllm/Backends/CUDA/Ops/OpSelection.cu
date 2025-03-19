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
      _warp_level_safe_softmax_fp32<16, 2, 32>
          <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
      return;
    }

    // CASE 1.3:
    // cols <= 256
    dim3 block_dims(8, 8);               // 64 threads per block
    dim3 grid_dims((rows + 8 - 1) / 8);  // ceil_div(rows, 8)
    _warp_level_safe_softmax_fp32<8, 2, 32>
        <<<grid_dims, block_dims>>>((float*)Z, (float*)X, rows, cols);
    return;
  }
}

}  // namespace mllm::cuda