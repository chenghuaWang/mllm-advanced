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

namespace mllm::cuda {

// see:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
constexpr int MLLM_CUDA_WARP_NUM = 32;
constexpr int MLLM_CUDA_PER_BLOCK_MAX_THREAD_NUM = 1024;
constexpr int MLLM_CUDA_PER_SM_32BIT_REGISTER_NUM = 64 * 1024;
constexpr int MLLM_CUDA_PER_THREAD_32BIT_REGISTER_NUM = 255;

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

}  // namespace mllm::cuda