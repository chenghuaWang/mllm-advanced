/**
 * @file LLMEmbeddingTokenOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/LLMEmbeddingTokenOp.hpp"

namespace mllm::arm {

ArmLLMEmbeddingTokenOp::ArmLLMEmbeddingTokenOp(const LLMEmbeddingTokenOpCargo& cargo)
    : LLMEmbeddingTokenOp(cargo) {}

void ArmLLMEmbeddingTokenOp::forward(const std::vector<Tensor>& inputs,
                                     std::vector<Tensor>& outputs) {
  // TODO
}

}  // namespace mllm::arm
