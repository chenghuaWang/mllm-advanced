/**
 * @file RMSNorm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/RMSNormOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

RMSNormOp::RMSNormOp(const RMSNormOpCargo& cargo) : BaseOp(OpType::kRMSNorm), cargo_(cargo) {}

void RMSNormOp::load(std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
}

void RMSNormOp::trace(void* trace_context, std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  MLLM_WARN("RMSNormOp::trace is not implemented");
}

void RMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("RMSNormOp::forward is not implemented");
}

void RMSNormOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(output_0);
}

void RMSNormOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
