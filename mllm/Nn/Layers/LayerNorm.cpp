/**
 * @file LayerNorm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/LayerNorm.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::nn {

LayerNorm::LayerNorm() : Layer(OpType::kLayerNorm, LayerNormOpCargo{}) {}

LayerNorm::LayerNorm(int32_t normalized_shape, bool elementwise_affine, bool bias, float eps)
    : Layer(OpType::kLayerNorm, LayerNormOpCargo{.normalized_shape = {normalized_shape},
                                                 .elementwise_affine = elementwise_affine,
                                                 .bias = bias,
                                                 .eps = eps}) {}

LayerNorm::LayerNorm(const std::vector<int32_t>& normalized_shape, bool elementwise_affine,
                     bool bias, float eps)
    : Layer(OpType::kLayerNorm, LayerNormOpCargo{.normalized_shape = normalized_shape,
                                                 .elementwise_affine = elementwise_affine,
                                                 .bias = bias,
                                                 .eps = eps}) {}
LayerNorm::LayerNorm(const LayerNormOpCargo& cargo) : Layer(OpType::kLayerNorm, cargo) {}

Tensor LayerNorm::weight() const {
  return Tensor(impl()->refParams()[impl()->absoluteName() + ".weight"]);
}

Tensor LayerNorm::bias() const {
  auto bias_name = impl()->absoluteName() + ".bias";
  if (!impl()->refParams().count(bias_name)) {
    MLLM_ERROR("There is no bias in the LayerNorm layer: {}", impl()->absoluteName());
    return Tensor(nullptr);
  }
  return Tensor(impl()->refParams()[bias_name]);
}

}  // namespace mllm::nn
