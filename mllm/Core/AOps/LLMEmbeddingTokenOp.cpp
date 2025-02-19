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

namespace mllm {

LLMEmbeddingTokenOp::LLMEmbeddingTokenOp(const LLMEmbeddingTokenOpCargo& cargo)
    : BaseOp(OpType::kLLMEmbeddingToken), cargo_(cargo) {}

void LLMEmbeddingTokenOp::load(std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
}

void LLMEmbeddingTokenOp::trace(void* trace_context, std::vector<Tensor>& inputs,
                                std::vector<Tensor>& outputs) {
  NYI("LLMEmbeddingTokenOp::trace is not implemented");
}

void LLMEmbeddingTokenOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LLMEmbeddingTokenOp::forward is not implemented");
}

void LLMEmbeddingTokenOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto i = inputs[0];
  auto shape = i.shape();
  std::vector<size_t> o_shape{/*batch*/ shape[0], /*seq*/ shape[1],
                              /*feat dim*/ (size_t)cargo_.hidden_size};
  outputs.emplace_back(Tensor::empty(o_shape, i.dtype(), i.device()));
}

void LLMEmbeddingTokenOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
