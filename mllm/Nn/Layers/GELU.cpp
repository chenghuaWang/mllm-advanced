/**
 * @file GELU.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/GELU.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::nn {

GELU::GELU() : Layer(OpType::kGELU, GELUOpCargo{}) {}

GELU::GELU(const GELUOpCargo& cargo) : Layer(OpType::kGELU, cargo) {}

}  // namespace mllm::nn
