/**
 * @file Matmul.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <array>
#include "mllm/Core/Tensor.hpp"
#include "mllm/Core/AOps/SplitOp.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm::nn::F {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A = false, bool transpose_B = false);

Tensor view(const Tensor& x, const std::vector<int32_t>& shape);

std::vector<Tensor> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim);

std::vector<Tensor> split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections,
                          int32_t dim);

// For structure binding usage. But will increase compile time.
// e.g.:
// Tensor x = Tensor::ones({10, 2, 1024}, kFp32, kCPU);
// auto [x1, x2, x3, x4] = split<4>(x, 256, -1);
// assert(x1.shape()[2] == 1024 / 4)
// assert(x2.shape()[2] == 1024 / 4)
// assert(x3.shape()[2] == 1024 / 4)
// assert(x4.shape()[2] == 1024 / 4)
template<int32_t RET_NUM>
std::array<Tensor, RET_NUM> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim) {
  auto outputs = MllmEngineCtx::instance().dispatch(
      OpType::kSplit,
      SplitOpCargo{.dim_ = dim, .split_size_or_sections_ = {split_size_or_sections}}, {x});
  std::array<Tensor, RET_NUM> ret;

#pragma unroll
  for (int i = 0; i < RET_NUM; ++i) { ret[i] = outputs[i]; }

  return ret;
}

template<int32_t RET_NUM>
std::array<Tensor, RET_NUM> split(const Tensor& x,
                                  const std::vector<int32_t>& split_size_or_sections, int32_t dim) {
  auto outputs = MllmEngineCtx::instance().dispatch(
      OpType::kSplit, SplitOpCargo{.dim_ = dim, .split_size_or_sections_ = split_size_or_sections},
      {x});
  std::array<Tensor, RET_NUM> ret;

#pragma unroll
  for (int i = 0; i < RET_NUM; ++i) { ret[i] = outputs[i]; }

  return ret;
}

Tensor concat(const std::vector<Tensor>& ins, int32_t dim);

}  // namespace mllm::nn::F
