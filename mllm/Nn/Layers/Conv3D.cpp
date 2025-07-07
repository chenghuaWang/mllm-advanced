/**
 * @file Conv3D.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/Conv3D.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::nn {

Conv3D::Conv3D() : Layer(OpType::kConv3D, Conv3DOpCargo{}) {}

Conv3D::Conv3D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size,
               const std::vector<int32_t>& stride_size, bool bias, Conv3DOpImplType impl_type)
    : Layer(OpType::kConv3D, Conv3DOpCargo{
                                 .in_channels = in_channels,
                                 .out_channels = out_channels,
                                 .kernel_size = kernel_size,
                                 .stride = stride_size,
                                 .bias = bias,
                                 .impl_type = impl_type,
                             }) {}

Conv3D::Conv3D(const Conv3DOpCargo& cargo) : Layer(OpType::kConv3D, cargo) {}

Tensor Conv3D::weight() const {
  return Tensor(impl()->refParams()[impl()->absoluteName() + ".weight"]);
}

Tensor Conv3D::bias() const {
  auto bias_name = impl()->absoluteName() + ".bias";
  if (!impl()->refParams().count(bias_name)) {
    MLLM_ERROR("There is no bias in the Conv3D layer: {}", impl()->absoluteName());
    return Tensor(nullptr);
  }
  return Tensor(impl()->refParams()[bias_name]);
}

}  // namespace mllm::nn
