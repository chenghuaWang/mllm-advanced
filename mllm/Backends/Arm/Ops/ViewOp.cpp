/**
 * @file ViewOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/ViewOp.hpp"

namespace mllm::arm {

ArmViewOp::ArmViewOp(const ViewOpCargo& cargo) : ViewOp(cargo) {}

}  // namespace mllm::arm