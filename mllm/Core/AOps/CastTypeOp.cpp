/**
 * @file CastTypeOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/CastTypeOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Tensor/Value.hpp"

namespace mllm {

CastTypeOp::CastTypeOp(const CastTypeOpCargo& cargo) : BaseOp(OpType::kCastType), cargo_(cargo) {}

void CastTypeOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void CastTypeOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                       std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::CastTypeOp>(shared_from_this(), i_irs, o_irs);
}

void CastTypeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("CastTypeOp::forward is not implemented");
}

void CastTypeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto o = Tensor::empty(inputs[0].shape(), cargo_.to_dtype, inputs[0].device());
  outputs.emplace_back(o);
}

void CastTypeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
