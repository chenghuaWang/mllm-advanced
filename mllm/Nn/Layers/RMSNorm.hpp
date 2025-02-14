/**
 * @file RMSNorm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/RMSNormOp.hpp"

namespace mllm::nn {

class RMSNorm : public Layer {
 public:
  RMSNorm();

  RMSNorm(float eps, bool add_unit_offset = false);

  explicit RMSNorm(const RMSNormOpCargo& cargo);

  [[nodiscard]] Tensor weight() const;
};

}  // namespace mllm::nn
