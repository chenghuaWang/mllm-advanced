/**
 * @file RMSNorm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/RMSNormOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

RMSNormOp::RMSNormOp(const RMSNormOpCargo& cargo) : BaseOp(OpType::kRMSNorm), cargo_(cargo) {}

void RMSNormOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
}

void RMSNormOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::RMSNormOp>(shared_from_this(), i_irs, o_irs);

  // Save parameters to global look up table
  for (auto& p : this->params()) {
    MLLM_RT_ASSERT_EQ(p.second.name(), p.first);
    auto v = ctx->create<ir::tensor::TensorValue>(p.second);
    v->name() = p.first;
    ctx->addToSymbolTable(v, p.first);
  }
}

void RMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("RMSNormOp::forward is not implemented");
}

void RMSNormOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(output_0);
}

void RMSNormOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

RMSNormOp::params_t RMSNormOp::params() {
  params_t ret;
  ret.insert({name() + ".weight", weight_});
  return ret;
}

}  // namespace mllm
