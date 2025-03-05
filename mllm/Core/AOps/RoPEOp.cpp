/**
 * @file RoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/RoPEOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {
RoPEOp::RoPEOp(const RoPEOpCargo& cargo) : BaseOp(OpType::kRoPE), cargo_(cargo) {}

void RoPEOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  MLLM_WARN("RoPEOp::load is not implemented");
}

void RoPEOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                   std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::RoPEOp>(this, i_irs, o_irs);
}

void RoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("RoPEOp::forward is not implemented");
}

void RoPEOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(output_0);
}

void RoPEOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
