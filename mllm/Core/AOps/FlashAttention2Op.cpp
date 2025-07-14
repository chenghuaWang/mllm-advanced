/**
 * @file FlashAttention2Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-05-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/FlashAttention2Op.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {
FlashAttn2Op::FlashAttn2Op(const FlashAttn2OpCargo& cargo)
    : BaseOp(OpType::kFlashAttention_2), cargo_(cargo) {}

void FlashAttn2Op::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void FlashAttn2Op::trace(void* trace_context, const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) {
  MLLM_WARN("FlashAttn2Op::trace is not implemented.");
}

void FlashAttn2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("FlashAttn2Op::forward is not implemented");
}

void FlashAttn2Op::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i = inputs[0];

  // suppose inputs[0] is query
  outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
}

void FlashAttn2Op::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}
}  // namespace mllm
