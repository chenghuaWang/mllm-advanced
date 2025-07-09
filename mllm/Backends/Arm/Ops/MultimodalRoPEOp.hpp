/**
 * @file MultimodalRoPEOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/MultimodalRoPEOp.hpp"

namespace mllm::arm {

struct Qwen2VLMultimodalRoPEOpImpl {
  Tensor makeInvFreq(int output_dim, float rope_theta);

  std::pair<Tensor, Tensor> makePositionEmbedding(Tensor& position_ids, Tensor& inv_freq,
                                                  int seq_len, int output_dim,
                                                  std::vector<int32_t>& mrope_section);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin,
               Tensor& cos);
};

class ArmMultimodalRoPEOp final : public MultimodalRoPEOp {
 public:
  explicit ArmMultimodalRoPEOp(const MultimodalRoPEOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmMultimodalRoPEOpFactory
    : public TypedOpFactory<OpType::kMultimodalRoPE, MultimodalRoPEOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const MultimodalRoPEOpCargo& cargo) override {
    return std::make_shared<ArmMultimodalRoPEOp>(cargo);
  }
};

}  // namespace mllm::arm