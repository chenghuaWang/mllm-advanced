/**
 * @file LLMEmbeddingToken.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/LLMEmbeddingTokenOp.hpp"

namespace mllm::nn {

class LLMEmbeddingToken : public Layer {
 public:
  LLMEmbeddingToken();

  explicit LLMEmbeddingToken(const LLMEmbeddingTokenOpCargo& cargo);

  LLMEmbeddingToken(int vocab_size, int hidden_size);
};

}  // namespace mllm::nn