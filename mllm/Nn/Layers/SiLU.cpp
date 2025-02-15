/**
 * @file SiLU.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/SiLU.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::nn {

SiLU::SiLU() : Layer(OpType::kSiLU, SiLUOpCargo{}) {}

SiLU::SiLU(const SiLUOpCargo& cargo) : Layer(OpType::kSiLU, cargo) {}

}  // namespace mllm::nn
