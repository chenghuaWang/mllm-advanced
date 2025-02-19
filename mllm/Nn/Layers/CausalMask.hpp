/**
 * @file CausalMask.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/CausalMaskOp.hpp"

namespace mllm::nn {

class CausalMask : public Layer {
 public:
  CausalMask();
  explicit CausalMask(const CausalMaskOpCargo& cargo);
};

}  // namespace mllm::nn