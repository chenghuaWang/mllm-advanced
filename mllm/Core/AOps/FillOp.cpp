/**
 * @file FillOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/FillOp.hpp"

namespace mllm {

FillOp::FillOp(const FillOpCargo& cargo) : BaseOp(OpType::kFill), cargo_(cargo) {}

void FillOp::load(std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void FillOp::trace(void* trace_contex, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("FillOp::trace is not implemented");
}

void FillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("FillOp::forward is not implemented");
}

void FillOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // replace, using inputs' memory space
  outputs.emplace_back(inputs[0]);
}

void FillOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // do nothing.
}

}  // namespace mllm
