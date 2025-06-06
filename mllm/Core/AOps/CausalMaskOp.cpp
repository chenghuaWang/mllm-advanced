/**
 * @file CausalMaskOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/CausalMaskOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {
CausalMaskOp::CausalMaskOp(const CausalMaskOpCargo& cargo)
    : BaseOp(OpType::kCausalMask), cargo_(cargo) {}

void CausalMaskOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void CausalMaskOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::CausalMaskOp>(shared_from_this(), i_irs, o_irs);
}

void CausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("CausalMaskOp::forward is not implemented");
}

void CausalMaskOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
}

void CausalMaskOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}
}  // namespace mllm
