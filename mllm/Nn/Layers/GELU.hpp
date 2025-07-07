/**
 * @file GELU.hpp
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
#include "mllm/Core/AOps/GELUOp.hpp"

namespace mllm::nn {

class GELU : public Layer {
 public:
  GELU();

  explicit GELU(const GELUOpCargo& cargo);
};

}  // namespace mllm::nn
