/**
 * @file Linear.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/Linear.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::nn {

Linear::Linear() : Layer(OpType::kLinear, LinearOpCargo{}) {}

Linear::Linear(int32_t in_channels, int32_t out_channels, bool bias, bool transpose,
               LinearOpImplType impl_type)
    : Layer(OpType::kLinear, LinearOpCargo{.in_channels = in_channels,
                                           .out_channels = out_channels,
                                           .bias = bias,
                                           .transpose = transpose,
                                           .impl_type_ = impl_type}) {}

Linear::Linear(const LinearOpCargo& cargo) : Layer(OpType::kLinear, cargo) {}

Tensor Linear::weight() const {
  return Tensor(impl()->refParams()[impl()->absoluteName() + ".weight"]);
}

Tensor Linear::bias() const {
  auto bias_name = impl()->absoluteName() + ".bias";
  if (!impl()->refParams().count(bias_name)) {
    MLLM_ERROR("There is no bias in the linear layer: {}", impl()->absoluteName());
    return Tensor(nullptr);
  }
  return Tensor(impl()->refParams()[bias_name]);
}

}  // namespace mllm::nn
