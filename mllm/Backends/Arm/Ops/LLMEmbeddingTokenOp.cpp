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
#include <cstring>

namespace mllm::arm {

ArmLLMEmbeddingTokenOp::ArmLLMEmbeddingTokenOp(const LLMEmbeddingTokenOpCargo& cargo)
    : LLMEmbeddingTokenOp(cargo) {}

void ArmLLMEmbeddingTokenOp::forward(const std::vector<Tensor>& inputs,
                                     std::vector<Tensor>& outputs) {
  auto ins = inputs[0];
  auto ous = outputs[0];

  auto B = ins.shape()[0];
  auto S = ins.shape()[1];

  auto weight_dtype = weight_.dtype();

  for (size_t b = 0; b < B; ++b) {
    for (size_t s = 0; s < S; ++s) {
      switch (weight_dtype) {
        case kFp32:
          std::memcpy(
              ous.offsettedRawPtr({b, s, 0}),
              weight_.ptr<float>() + cargo_.hidden_size * (*ins.offsettedPtr<int64_t>({b, s})),
              cargo_.hidden_size * sizeof(float));
          break;
        case kFp16:
        default: NYI("Not supported weight dtype for arm llm embedding token op");
      }
    }
  }
}

}  // namespace mllm::arm
