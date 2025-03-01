/**
 * @file D2HOp.cu
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/CUDACommons.hpp"
#include "mllm/Backends/CUDA/Ops/D2HOp.cuh"
#include "mllm/Core/DataTypes.hpp"

namespace mllm::cuda {

CUDAD2HOp::CUDAD2HOp(const D2HOpCargo& cargo) : D2HOp(cargo) {}

void CUDAD2HOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_CHECK_CUDA_ERROR(cudaMemcpy(outputs[0].ptr<char>(), inputs[0].ptr<char>(),
                                   inputs[0].numel() * dataTypeSize(inputs[0].dtype()),
                                   cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

}  // namespace mllm::cuda
