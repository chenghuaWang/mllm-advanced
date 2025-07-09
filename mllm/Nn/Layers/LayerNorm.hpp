/**
 * @file LayerNorm.hpp
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
#include "mllm/Core/AOps/LayerNormOp.hpp"

namespace mllm::nn {

class LayerNorm : public Layer {
 public:
  LayerNorm();

  explicit LayerNorm(int32_t normalized_shape, bool elementwise_affine, bool bias,
                     float eps = 1e-6);

  explicit LayerNorm(const std::vector<int32_t>& normalized_shape, bool elementwise_affine,
                     bool bias, float eps = 1e-6);

  explicit LayerNorm(const LayerNormOpCargo& cargo);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;
};

}  // namespace mllm::nn
