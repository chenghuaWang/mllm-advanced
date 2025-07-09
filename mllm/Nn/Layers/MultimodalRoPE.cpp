/**
 * @file MultimodalRoPE.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/MultimodalRoPE.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/MultimodalRoPEOp.hpp"

namespace mllm::nn {

MultimodalRoPE::MultimodalRoPE()
    : Layer(OpType::kMultimodalRoPE,
            MultimodalRoPEOpCargo{.type = MultimodalRoPEOpCargoType::kDefault}) {}

MultimodalRoPE::MultimodalRoPE(const MultimodalRoPEOpCargo& cargo)
    : Layer(OpType::kMultimodalRoPE, cargo) {}

MultimodalRoPE::MultimodalRoPE(MultimodalRoPEOpCargoType type, float rope_theta,
                               int32_t max_position_embeddings,
                               const std::vector<int32_t>& mrope_section)
    : Layer(OpType::kMultimodalRoPE,
            MultimodalRoPEOpCargo{.type = type,
                                  .rope_theta = rope_theta,
                                  .max_position_embeddings = max_position_embeddings,
                                  .mrope_section = mrope_section}) {}

}  // namespace mllm::nn