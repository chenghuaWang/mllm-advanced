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

namespace mllm::nn {
Linear::Linear(LinearOpCargo cargo) : cargo_(std::move(cargo)) {}

Tensor Linear::weight() const {
  return Tensor(impl()->refParams()[impl()->absoluteName() + ".weight"]);
}

Tensor Linear::bisa() const {
  auto bias_name = impl()->absoluteName() + ".bias";
  if (!impl()->refParams().count(bias_name)) {
    MLLM_ERROR("There is no bias in the linear layer: {}", impl()->absoluteName());
    return Tensor(nullptr);
  }
  return Tensor(impl()->refParams()[bias_name]);
}

}  // namespace mllm::nn
