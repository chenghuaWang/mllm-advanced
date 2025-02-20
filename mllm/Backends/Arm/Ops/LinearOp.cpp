/**
 * @file LinearOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/LinearOp.hpp"
#include "mllm/Core/AOps/LinearOp.hpp"

namespace mllm::arm {

ArmLinearOp::ArmLinearOp(const LinearOpCargo& cargo) : LinearOp(cargo) {}

void ArmLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

}  // namespace mllm::arm
