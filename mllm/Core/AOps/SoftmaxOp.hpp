/**
 * @file Softmax.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct SoftmaxOpCargo : public BaseOpCargo<SoftmaxOpCargo> {
  int axis;
};

class SoftmaxOp : public BaseOp {
 public:
  explicit SoftmaxOp(const SoftmaxOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  SoftmaxOpCargo cargo_;
};

}  // namespace mllm
