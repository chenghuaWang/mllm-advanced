/**
 * @file D2HOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/D2HOp.hpp"

namespace mllm {

D2HOp::D2HOp(const D2HOpCargo& cargo) : BaseOp(OpType::kD2H), cargo_(cargo) {}

void D2HOp::load(std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void D2HOp::trace(void* trace_context, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("D2HOp::trace is not implemented");
}

void D2HOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("D2HOp::forward is not implemented");
}

void D2HOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto o = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), cargo_.to_device_type);
  outputs.emplace_back(o);
}

void D2HOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
