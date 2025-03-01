/**
 * @file ElewiseOps.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/Ops/ElewiseOps.cuh"
#include "mllm/Backends/CUDA/Kernels/elewise.cuh"

namespace mllm::cuda {

void CUDAAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto X = inputs[0];
  auto Y = inputs[1];
  auto Z = outputs[0];
  auto size = X.numel();

  if (X.dtype() == kBF16 && Y.dtype() == kBF16 && Z.dtype() == kBF16) {
    int block_size = 1024;
    int grid = (size + block_size * 8 - 1) / (block_size * 8);
    vector_add_bf16_v0<8><<<grid, block_size>>>(Z.ptr<nv_bfloat16>(), X.ptr<nv_bfloat16>(),
                                                Y.ptr<nv_bfloat16>(), size, 1.0f, 1.0f, 0.0f);
  }
}

}  // namespace mllm::cuda
