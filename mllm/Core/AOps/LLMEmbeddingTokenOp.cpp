/**
 * @file LLMEmbeddingTokenOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/LLMEmbeddingTokenOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

LLMEmbeddingTokenOp::LLMEmbeddingTokenOp(const LLMEmbeddingTokenOpCargo& cargo)
    : BaseOp(OpType::kLLMEmbeddingToken), cargo_(cargo) {}

void LLMEmbeddingTokenOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
}

void LLMEmbeddingTokenOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                                std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::LLMEmbeddingTokenOp>(shared_from_this(), i_irs, o_irs);
}

void LLMEmbeddingTokenOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LLMEmbeddingTokenOp::forward is not implemented");
}

void LLMEmbeddingTokenOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto shape = i.shape();
  std::vector<int32_t> o_shape{/*batch*/ shape[0], /*seq*/ shape[1],
                               /*feat dim*/ cargo_.hidden_size};
  outputs.emplace_back(Tensor::empty(o_shape, weight_.dtype(), i.device()));
}

void LLMEmbeddingTokenOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

LLMEmbeddingTokenOp::params_t LLMEmbeddingTokenOp::params() {
  params_t ret;
  ret.insert({name() + ".weight", weight_});
  return ret;
}

}  // namespace mllm
