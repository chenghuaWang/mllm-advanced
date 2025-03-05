/**
 * @file FillOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

// type
// 0 -> zeros
// 1 -> ones
// 2 -> specific
// 3 -> random
// 4 -> arrange
// 5 -> make input tensor contiguous
struct FillOpCargo : public BaseOpCargo<FillOpCargo> {
  size_t type = 0;
  float value = 0.f;
  float start = 0.f;
  float end = 0.f;
  float step = 0.f;
};

class FillOp : public BaseOp {
 public:
  explicit FillOp(const FillOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  FillOpCargo cargo_;
};

}  // namespace mllm
