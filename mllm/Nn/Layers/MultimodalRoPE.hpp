/**
 * @file MultimodalRoPE.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/MultimodalRoPEOp.hpp"

namespace mllm::nn {

class MultimodalRoPE : public Layer {
 public:
  MultimodalRoPE();

  explicit MultimodalRoPE(const MultimodalRoPEOpCargo& cargo);

  MultimodalRoPE(MultimodalRoPEOpCargoType type, float rope_theta, int32_t max_position_embeddings,
                 const std::vector<int32_t>& mrope_section);
};

}  // namespace mllm::nn
