/**
 * @file Softmax.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/SoftmaxOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

SoftmaxOp::SoftmaxOp(const SoftmaxOpCargo& cargo) : BaseOp(OpType::kSoftmax), cargo_(cargo) {}

void SoftmaxOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void SoftmaxOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::SoftmaxOp>(this, i_irs, o_irs);
}

void SoftmaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("SoftmaxOp::forward is not implemented");
}

void SoftmaxOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs.emplace_back(Tensor(
      std::make_shared<TensorImpl>(inputs[0].shape(), inputs[0].dtype(), inputs[0].device())));
}

void SoftmaxOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
