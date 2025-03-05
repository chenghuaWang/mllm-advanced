/**
 * @file RMSNorm.hpp
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

struct RMSNormOpCargo : public BaseOpCargo<RMSNormOpCargo> {
  float epsilon;

  // for Gemma
  bool add_unit_offset = false;
};

class RMSNormOp : public BaseOp {
 public:
  explicit RMSNormOp(const RMSNormOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  Tensor weight_;
  RMSNormOpCargo cargo_;
};

}  // namespace mllm
