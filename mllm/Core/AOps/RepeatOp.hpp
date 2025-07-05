/**
 * @file RepeatOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {
struct RepeatOpCargo : public BaseOpCargo<RepeatOpCargo> {
  int32_t multiplier;
  int32_t dim;
};

class RepeatOp : public BaseOp {
 public:
  explicit RepeatOp(const RepeatOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline int32_t multiplier() const { return cargo_.multiplier; }

  inline int32_t dim() const { return cargo_.dim; }

 protected:
  RepeatOpCargo cargo_;
};

}  // namespace mllm
