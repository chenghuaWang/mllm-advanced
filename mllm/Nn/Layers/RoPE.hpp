/**
 * @file RoPE.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/RoPEOp.hpp"

namespace mllm::nn {

class RoPE : public Layer {
 public:
  RoPE();

  explicit RoPE(const RoPEOpCargo& cargo);

  RoPE(RoPETypes type, float theta, int max_position_embeddings, int dims);

  RoPE(RoPETypes type, float theta, int max_position_embeddings, int dims,
       RoPEOpCargo::RoPELayoutType layout_type);
};

}  // namespace mllm::nn