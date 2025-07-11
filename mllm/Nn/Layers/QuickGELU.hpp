/**
 * @file QuickGELU.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/QuickGELUOp.hpp"

namespace mllm::nn {

class QuickGELU : public Layer {
 public:
  QuickGELU();
  explicit QuickGELU(const QuickGELUOpCargo& cargo);
};

}  // namespace mllm::nn