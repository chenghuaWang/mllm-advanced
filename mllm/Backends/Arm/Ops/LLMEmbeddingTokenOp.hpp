/**
 * @file LLMEmbeddingTokenOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/LLMEmbeddingTokenOp.hpp"

namespace mllm::arm {

class ArmLLMEmbeddingTokenOp final : public LLMEmbeddingTokenOp {
 public:
  explicit ArmLLMEmbeddingTokenOp(const LLMEmbeddingTokenOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmLLMEmbeddingTokenOpFactory
    : public TypedOpFactory<OpType::kLLMEmbeddingToken, LLMEmbeddingTokenOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const LLMEmbeddingTokenOpCargo& cargo) override {
    return std::make_shared<ArmLLMEmbeddingTokenOp>(cargo);
  }
};

}  // namespace mllm::arm