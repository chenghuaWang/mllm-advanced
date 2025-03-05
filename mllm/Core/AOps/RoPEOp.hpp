/**
 * @file RoPEOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

enum class RoPETypes : int32_t {
  kRoPETypes_Start = 0,
  kLlama2,
  kRoPETypes_End,
};

struct RoPEOpCargo : public BaseOpCargo<RoPEOpCargo> {
  RoPETypes type = RoPETypes::kLlama2;
  float theta;
  int max_position_embeddings;
  int dims;
};

// The input should be in [B, H, S, D] layout.
class RoPEOp : public BaseOp {
 public:
  explicit RoPEOp(const RoPEOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  int cur_seq_cnt_ = 0;
  RoPEOpCargo cargo_;
};

}  // namespace mllm
