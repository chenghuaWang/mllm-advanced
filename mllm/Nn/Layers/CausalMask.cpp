/**
 * @file CausalMask.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/CausalMask.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Nn/Layer.hpp"

namespace mllm::nn {

CausalMask::CausalMask() : Layer(OpType::kCausalMask, CausalMaskOpCargo{}) {}

CausalMask::CausalMask(const CausalMaskOpCargo& cargo) : Layer(OpType::kCausalMask, cargo) {}

}  // namespace mllm::nn
