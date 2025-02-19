/**
 * @file F.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/F/F.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/MatMulOp.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm::nn::F {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A, bool transpose_B) {
  return MllmEngineCtx::instance().dispatch(
      OpType::kMatMul, MatMulOpCargo{.transpose_a = transpose_A, .transpose_b = transpose_B},
      {A, B})[0];
}

}  // namespace mllm::nn::F
