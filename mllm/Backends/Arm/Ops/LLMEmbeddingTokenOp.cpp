/**
 * @file LLMEmbeddingTokenOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#endif
#include "mllm/Backends/Arm/Ops/LLMEmbeddingTokenOp.hpp"
#include <cstring>
#include <arm_neon.h>

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

  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      switch (weight_dtype) {
        case kFp32:
          std::memcpy(
              ous.offsettedRawPtr({b, s, 0}),
              weight_.ptr<float>() + cargo_.hidden_size * (*ins.offsettedPtr<int64_t>({b, s})),
              cargo_.hidden_size * sizeof(float));
          break;
        case kFp16:
          std::memcpy(
              ous.offsettedRawPtr({b, s, 0}),
              weight_.ptr<float16_t>() + cargo_.hidden_size * (*ins.offsettedPtr<int64_t>({b, s})),
              cargo_.hidden_size * sizeof(float16_t));
          break;
        default: NYI("Not supported weight dtype for arm llm embedding token op");
      }
    }
  }
}

}  // namespace mllm::arm
