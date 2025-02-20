/**
 * @file MatMulOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/MatMulOp.hpp"

namespace mllm::arm {

ArmMatMulOp::ArmMatMulOp(const MatMulOpCargo& cargo) : MatMulOp(cargo) {}

void ArmMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

}  // namespace mllm::arm
