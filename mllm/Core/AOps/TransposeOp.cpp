/**
 * @file TransposeOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/TransposeOp.hpp"
#include <utility>
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

TransposeOp::TransposeOp(const TransposeOpCargo& cargo)
    : BaseOp(OpType::kTranspose), cargo_(cargo) {}

void TransposeOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void TransposeOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::TransposeOp>(shared_from_this(), i_irs, o_irs);
}

void TransposeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("TransposeOp::forward is not implemented");
}

void TransposeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto shape = inputs[0].shape();
  std::swap(shape[cargo_.transpose_dim_x], shape[cargo_.transpose_dim_y]);
  Tensor o = Tensor::empty(shape, inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(o);
}

void TransposeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
