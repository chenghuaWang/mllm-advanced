/**
 * @file VisionRoPE.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/VisionRoPEOp.hpp"

namespace mllm::nn {

class VisionRoPE : public Layer {
 public:
  VisionRoPE();

  explicit VisionRoPE(const VisionRoPEOpCargoType type, const Qwen2VLRoPEOpCargo& cargo);

  explicit VisionRoPE(const VisionRoPEOpCargo& cargo);
};

}  // namespace mllm::nn
