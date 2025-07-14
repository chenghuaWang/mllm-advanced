/**
 * @file VisionRoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/VisionRoPEOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

VisionRoPEOp::VisionRoPEOp(const VisionRoPEOpCargo& cargo)
    : BaseOp(OpType::kVisionRoPE), cargo_(cargo) {}

void VisionRoPEOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void VisionRoPEOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::VisionRoPEOp>(shared_from_this(), i_irs, o_irs);
}

void VisionRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("VisionRoPEOp::forward is not implemented");
}

void VisionRoPEOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void VisionRoPEOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm