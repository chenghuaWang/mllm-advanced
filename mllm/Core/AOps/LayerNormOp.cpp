/**
 * @file LayerNormOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/LayerNormOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

LayerNormOp::LayerNormOp(const LayerNormOpCargo& cargo)
    : BaseOp(OpType::kLayerNorm), cargo_(cargo) {}

void LayerNormOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  if (cargo_.elementwise_affine) { weight_ = Tensor(ploader->operator[](name() + ".weight")); }
  if (cargo_.bias) { weight_ = Tensor(ploader->operator[](name() + ".bias")); }
}

void LayerNormOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::LayerNormOp>(shared_from_this(), i_irs, o_irs);
}

void LayerNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("LayerNormOp::forward is not implemented");
}

void LayerNormOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void LayerNormOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

LayerNormOp::params_t LayerNormOp::params() {
  params_t ret;
  if (cargo_.elementwise_affine) { ret.insert({name() + ".weight", weight_}); }
  if (cargo_.bias) { ret.insert({name() + ".bias", bias_}); }
  return ret;
}

}  // namespace mllm
