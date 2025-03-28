/**
 * @file SplitOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/SplitOp.hpp"

namespace mllm {

ArmSplitOp::ArmSplitOp(const SplitOpCargo& cargo) : SplitOp(cargo) {}

}  // namespace mllm
