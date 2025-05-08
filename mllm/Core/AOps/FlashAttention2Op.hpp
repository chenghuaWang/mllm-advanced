/**
 * @file FlashAttention2Op.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-05-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct FlashAttn2OpCargo : public BaseOpCargo<FlashAttn2OpCargo> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D;
  int32_t threads = 4;
  bool hp_exp = false;
  bool causal_mask = true;
};

class FlashAttn2Op : public BaseOp {
 public:
  explicit FlashAttn2Op(const FlashAttn2OpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  FlashAttn2OpCargo cargo_;
};

}  // namespace mllm
