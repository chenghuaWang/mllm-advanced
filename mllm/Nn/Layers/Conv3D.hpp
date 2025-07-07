/**
 * @file Conv3D.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/Conv3DOp.hpp"

namespace mllm::nn {

class Conv3D : public Layer {
 public:
  Conv3D();

  Conv3D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size,
         const std::vector<int32_t>& stride_size, bool bias = true,
         Conv3DOpImplType impl_type = Conv3DOpImplType::kDefault);

  explicit Conv3D(const Conv3DOpCargo& cargo);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;
};

}  // namespace mllm::nn
