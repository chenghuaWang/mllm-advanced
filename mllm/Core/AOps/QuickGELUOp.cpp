/**
 * @file QuickGELUOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/QuickGELUOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

QuickGELUOp::QuickGELUOp(const QuickGELUOpCargo& cargo)
    : BaseOp(OpType::kQuickGELU), cargo_(cargo) {}

void QuickGELUOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void QuickGELUOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::QuickGELUOp>(shared_from_this(), i_irs, o_irs);
}

void QuickGELUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("QuickGELUOp::forward is not implemented");
}

void QuickGELUOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void QuickGELUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm
