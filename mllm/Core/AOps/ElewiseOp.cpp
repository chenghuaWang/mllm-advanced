/**
 * @file ElewiseOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/ElewiseOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm {

AddOp::AddOp() : BaseOp(OpType::kAdd) {}

void AddOp::load(void* params) { MLLM_WARN("AddOp::load is not implemented"); }

void AddOp::trace(void* trace_contex, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("AddOp::trace is not implemented");
}

void AddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("AddOp::forward is not implemented");
}

void AddOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO I have not impl the broad cast yet.
  Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(output_0);
}

void AddOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  for (auto& t : outputs) t.alloc();
}

}  // namespace mllm
