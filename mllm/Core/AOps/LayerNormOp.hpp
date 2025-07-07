/**
 * @file LayerNormOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {
struct LayerNormOpCargo : public BaseOpCargo<LayerNormOpCargo> {
  int32_t dim;
  float eps = 1e-6;
  bool elementwise_affine = true;
  bool bias = true;
};

class LayerNormOp : public BaseOp {
 public:
  explicit LayerNormOp(const LayerNormOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  params_t params() override;

 protected:
  Tensor weight_;
  Tensor bias_;
  LayerNormOpCargo cargo_;
};

}  // namespace mllm