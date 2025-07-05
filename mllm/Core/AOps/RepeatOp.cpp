/**
 * @file RepeatOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "mllm/Core/AOps/RepeatOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {
RepeatOp::RepeatOp(const RepeatOpCargo& cargo) : BaseOp(OpType::kRepeat), cargo_(cargo) {}

void RepeatOp::load(const std::shared_ptr<ParameterLoader>& ploader) { MLLM_EMPTY_SCOPE; }

void RepeatOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::RepeatOp>(shared_from_this(), i_irs, o_irs);
}

void RepeatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("RepeatOp::forward is not implemented");
}

void RepeatOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();
  i_shape[cargo_.dim] *= cargo_.multiplier;
  outputs.emplace_back(Tensor::empty(i_shape, i.dtype(), i.device()));
}

void RepeatOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}
}  // namespace mllm