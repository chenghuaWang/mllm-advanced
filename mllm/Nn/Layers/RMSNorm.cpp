/**
 * @file RMSNorm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/RMSNorm.hpp"

namespace mllm::nn {

RMSNorm::RMSNorm() : Layer(OpType::kRMSNorm, RMSNormOpCargo{}) {}

RMSNorm::RMSNorm(float eps, bool add_unit_offset)
    : Layer(OpType::kRMSNorm, RMSNormOpCargo{.epsilon = eps, .add_unit_offset = add_unit_offset}) {}

RMSNorm::RMSNorm(const RMSNormOpCargo& cargo) : Layer(OpType::kRMSNorm, cargo) {}

Tensor RMSNorm::weight() const {
  return Tensor(impl()->refParams()[impl()->absoluteName() + ".weight"]);
}
}  // namespace mllm::nn
