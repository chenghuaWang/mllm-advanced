/**
 * @file LLMEmbeddingTokenOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct LLMEmbeddingTokenOpCargo : public BaseOpCargo<LLMEmbeddingTokenOpCargo> {
  int vocab_size = 0;
  int hidden_size = 0;
};

class LLMEmbeddingTokenOp : public BaseOp {
 public:
  explicit LLMEmbeddingTokenOp(const LLMEmbeddingTokenOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  Tensor weight_;
  LLMEmbeddingTokenOpCargo cargo_;
};

}  // namespace mllm
