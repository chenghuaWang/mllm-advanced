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

void vector_add_bf16_v0_call(void* Z, void* const X, void* const Y, int size, float a, float b,
                             float c) {
  int block_size = 1024;
  int grid = (size + block_size * 8 - 1) / (block_size * 8);
  vector_add_bf16_v0<8>
      <<<grid, block_size>>>((nv_bfloat16*)Z, (nv_bfloat16*)X, (nv_bfloat16*)Y, size, a, b, c);
  cudaError_t syncError = cudaGetLastError();
  if (syncError != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(syncError));
  }
}

}  // namespace mllm::cuda