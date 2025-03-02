/**
 * @file ElewiseOps.cu
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include "mllm/Backends/CUDA/CuProfiler.hpp"
#include "mllm/Backends/CUDA/Ops/OpSelection.hpp"

MLLM_CUDA_PROFILE_FUNC(vector_add_bf16, int iter_times, int size_0, int size_1) {
  std::string name = "vector_add_bf16_" + std::to_string(size_0) + "_" + std::to_string(size_1);
  mllm::cuda::CUDATimer timer;

  const float a = 1.0;
  const float b = 1.0;
  const float c = 0.0;

  size_t size = size_0 * size_1;

  void* cx = malloc(size * sizeof(nv_bfloat16));
  void* cy = malloc(size * sizeof(nv_bfloat16));
  void* cz = malloc(size * sizeof(nv_bfloat16));

  nv_bfloat16* gx;
  nv_bfloat16* gy;
  nv_bfloat16* gz;

  cudaMalloc(&gx, size * sizeof(nv_bfloat16));
  cudaMalloc(&gy, size * sizeof(nv_bfloat16));
  cudaMalloc(&gz, size * sizeof(nv_bfloat16));

  cudaMemcpy(gx, cx, size * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(gy, cy, size * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(gz, cz, size * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

  for (int i = 0; i < iter_times; i++) {
    mllm::cuda::ProfileScope profile_scope(profiler, name);
    mllm::cuda::vector_add_bf16_v0_call(gz, gx, gy, size_0 * size_1, a, b, c);
  }

  profiler.printReport();

  cudaFree(gx);
  cudaFree(gy);
  cudaFree(gz);

  free(cx);
  free(cy);
  free(cz);
}

int main() {
  std::vector<int> sizes = {1024, 2048};
  std::vector<mllm::cuda::CuProfiler> prefiles;

  for (auto& item : sizes) {
    mllm::cuda::CuProfiler p;
    MLLM_CUDA_PERF_vector_add_bf16(p, 100, item, item);
    prefiles.push_back(p);
  }
}
