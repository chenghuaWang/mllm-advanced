/**
 * @file RoPE.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/RoPE.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/RoPEOp.hpp"
#include "mllm/Nn/Layer.hpp"

namespace mllm::nn {

RoPE::RoPE() : Layer(OpType::kRoPE, RoPEOpCargo{}) {}

RoPE::RoPE(const RoPEOpCargo& cargo) : Layer(OpType::kRoPE, cargo) {}

RoPE::RoPE(RoPETypes type, float theta, int max_position_embeddings, int dims)
    : Layer(OpType::kRoPE, RoPEOpCargo{.type = type,
                                       .theta = theta,
                                       .max_position_embeddings = max_position_embeddings,
                                       .dims = dims}) {}

}  // namespace mllm::nn
