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
#include "mllm/Core/AOps/ConcatOp.hpp"
#include "mllm/Core/AOps/ViewOp.hpp"
#include "mllm/Core/AOps/SplitOp.hpp"
#include "mllm/Core/AOps/MatMulOp.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm::nn::F {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A, bool transpose_B) {
  return MllmEngineCtx::instance().dispatch(
      OpType::kMatMul, MatMulOpCargo{.transpose_a = transpose_A, .transpose_b = transpose_B},
      {A, B})[0];
}

Tensor view(const Tensor& x, const std::vector<int32_t>& shape) {
  return MllmEngineCtx::instance().dispatch(OpType::kView, ViewOpCargo{.to_shape_ = shape}, {x})[0];
}

std::vector<Tensor> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim) {
  return MllmEngineCtx::instance().dispatch(
      OpType::kSplit,
      SplitOpCargo{.dim_ = dim, .split_size_or_sections_ = {split_size_or_sections}}, {x});
}

std::vector<Tensor> split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections,
                          int32_t dim) {
  return MllmEngineCtx::instance().dispatch(
      OpType::kSplit, SplitOpCargo{.dim_ = dim, .split_size_or_sections_ = split_size_or_sections},
      {x});
}

Tensor concat(const std::vector<Tensor>& ins, int32_t dim) {
  return MllmEngineCtx::instance().dispatch(OpType::kConcat, ConcatOpCargo{.dim = dim}, ins)[0];
}

}  // namespace mllm::nn::F
