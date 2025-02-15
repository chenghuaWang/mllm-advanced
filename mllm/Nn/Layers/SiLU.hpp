/**
 * @file SiLU.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/SiLUOp.hpp"

namespace mllm::nn {

class SiLU : public Layer {
 public:
  SiLU();
  explicit SiLU(const SiLUOpCargo& cargo);
};

}  // namespace mllm::nn