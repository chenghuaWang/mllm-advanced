/**
 * @file LLMEmbeddingToken.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/LLMEmbeddingToken.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/LLMEmbeddingTokenOp.hpp"

namespace mllm::nn {

LLMEmbeddingToken::LLMEmbeddingToken()
    : Layer(OpType::kLLMEmbeddingToken, LLMEmbeddingTokenOpCargo{}) {}

LLMEmbeddingToken::LLMEmbeddingToken(const LLMEmbeddingTokenOpCargo& cargo)
    : Layer(OpType::kLLMEmbeddingToken, cargo) {}

LLMEmbeddingToken::LLMEmbeddingToken(int vocab_size, int hidden_size)
    : Layer(OpType::kLLMEmbeddingToken,
            LLMEmbeddingTokenOpCargo{.vocab_size = vocab_size, .hidden_size = hidden_size}) {}

}  // namespace mllm::nn
