/**
 * @file VisionRoPE.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/VisionRoPE.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/VisionRoPEOp.hpp"

namespace mllm::nn {

VisionRoPE::VisionRoPE() : Layer(OpType::kVisionRoPE, VisionRoPEOpCargo{}) {}

VisionRoPE::VisionRoPE(const VisionRoPEOpCargoType type, const Qwen2VLRoPEOpCargo& cargo)
    : Layer(OpType::kVisionRoPE, VisionRoPEOpCargo{
                                     .type = type,
                                     .qwen2vl_rope_op_cargo = cargo,
                                 }) {}

VisionRoPE::VisionRoPE(const VisionRoPEOpCargo& cargo) : Layer(OpType::kVisionRoPE, cargo) {}

}  // namespace mllm::nn
