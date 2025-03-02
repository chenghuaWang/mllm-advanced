/**
 * @file ElewiseOps.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/Ops/OpSelection.hpp"
#include "mllm/Backends/CUDA/Ops/ElewiseOps.hpp"

namespace mllm::cuda {

void CUDAAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto X = inputs[0];
  auto Y = inputs[1];
  auto Z = outputs[0];
  auto size = X.numel();

  if (X.dtype() == kBF16 && Y.dtype() == kBF16 && Z.dtype() == kBF16) {
    vector_add_bf16_v0_call(Z.ptr<char>(), X.ptr<char>(), Y.ptr<char>(), size, 1.0, 1.0, 0.0);
  }
}

}  // namespace mllm::cuda
