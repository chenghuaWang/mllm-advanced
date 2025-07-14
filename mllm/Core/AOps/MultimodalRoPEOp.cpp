/**
 * @file MultimodalRoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/MultimodalRoPEOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

MultimodalRoPEOp::MultimodalRoPEOp(const MultimodalRoPEOpCargo& cargo)
    : BaseOp(OpType::kMultimodalRoPE), cargo_(cargo) {}

void MultimodalRoPEOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void MultimodalRoPEOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                             std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::MultimodalRoPEOp>(shared_from_this(), i_irs, o_irs);
}

void MultimodalRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("MultimodalRoPEOp::forward is not implemented");
}

void MultimodalRoPEOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void MultimodalRoPEOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm
