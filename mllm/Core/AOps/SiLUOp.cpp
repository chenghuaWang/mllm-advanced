/**
 * @file SiLUOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/SiLUOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

SiLUOp::SiLUOp(const SiLUOpCargo& cargo) : BaseOp(OpType::kSiLU), cargo_(cargo) {}

void SiLUOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void SiLUOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                   std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::SiLUOp>(this, i_irs, o_irs);
}

void SiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("SiLUOp::forward is not implemented");
}

void SiLUOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto i = inputs[0];
  outputs.emplace_back(Tensor(std::make_shared<TensorImpl>(i.shape(), i.dtype(), i.device())));
}

void SiLUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
