/**
 * @file ViewOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct ViewOpCargo : public BaseOpCargo<ViewOpCargo> {
  std::vector<int32_t> to_shape_;
};

class ViewOp : public BaseOp {
 public:
  explicit ViewOp(const ViewOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline std::vector<int32_t> toWhichShape() { return cargo_.to_shape_; }

 protected:
  ViewOpCargo cargo_;
};

}  // namespace mllm
