/**
 * @file D2HOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DeviceTypes.hpp"

namespace mllm {
struct D2HOpCargo : public BaseOpCargo<D2HOpCargo> {
  DeviceTypes from_device_type;
  DeviceTypes to_device_type;
};

class D2HOp : public BaseOp {
 public:
  explicit D2HOp(const D2HOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  D2HOpCargo cargo_;
};

}  // namespace mllm