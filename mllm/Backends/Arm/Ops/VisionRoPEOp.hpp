/**
 * @file VisionRoPEOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/VisionRoPEOp.hpp"

namespace mllm::arm {

struct Qwen2VLVisionRoPEOpImpl {
  Tensor computeInvFreq(const Qwen2VLRoPEOpCargo& cargo);

  // Get Position ids.
  Tensor getRotaryPosEmbIds(Tensor& grid_thw, const Qwen2VLRoPEOpCargo& cargo);

  Tensor computeRotaryPosEmb(Tensor& rotary_pos_emb_full, Tensor& pos_ids, Tensor& grid_thw,
                             const Qwen2VLRoPEOpCargo& cargo);

  Tensor rotaryPosEmb(Tensor& inv_freq, int seq_len, const Qwen2VLRoPEOpCargo& cargo);

  std::pair<Tensor, Tensor> getSinCos(Tensor& rotary_pos_emb);

  void forward(const Tensor& activation, const Tensor& sin, const Tensor& cos, Tensor& out);
};

class ArmVisionRoPEOp final : public VisionRoPEOp {
 public:
  explicit ArmVisionRoPEOp(const VisionRoPEOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmVisionRoPEOpFactory : public TypedOpFactory<OpType::kVisionRoPE, VisionRoPEOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const VisionRoPEOpCargo& cargo) override {
    return std::make_shared<ArmVisionRoPEOp>(cargo);
  }
};

}  // namespace mllm::arm