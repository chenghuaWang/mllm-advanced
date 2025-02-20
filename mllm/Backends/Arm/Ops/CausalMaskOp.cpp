/**
 * @file CausalMaskOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/CausalMaskOp.hpp"

namespace mllm::arm {

ArmCausalMaskOp::ArmCausalMaskOp(const CausalMaskOpCargo& cargo) : CausalMaskOp(cargo) {}

void ArmCausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

}  // namespace mllm::arm