/**
 * @file CopyOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/CopyOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

CopyOp::CopyOp(const CopyOpCargo& cargo) : BaseOp(OpType::kCopy), cargo_(cargo) {}

void CopyOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void CopyOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                   std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::CopyOp>(shared_from_this(), i_irs, o_irs);
}

void CopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("CopyOp::forward is not implemented");
}

void CopyOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // do nothing.
}

void CopyOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // do nothing.
}

}  // namespace mllm
