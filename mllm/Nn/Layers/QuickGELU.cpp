/**
 * @file QuickGELU.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/QuickGELU.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::nn {

QuickGELU::QuickGELU() : Layer(OpType::kQuickGELU, QuickGELUOpCargo{}) {}

QuickGELU::QuickGELU(const QuickGELUOpCargo& cargo) : Layer(OpType::kQuickGELU, cargo) {}

}  // namespace mllm::nn
