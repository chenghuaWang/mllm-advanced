/**
 * @file ArmBackend.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/ArmBackend.hpp"
#include "mllm/Backends/Arm/ArmAllocator.hpp"
#include "mllm/Backends/Arm/Ops/ElewiseOps.hpp"

namespace mllm::arm {

ArmBackend::ArmBackend() : BackendBase(kCPU) {
  allocator_ = std::make_shared<ArmAllocator>();
  regOpFactory<ArmAddOpFactory, ArmSubOpFactory, ArmMulOpFactory, ArmDivOpFactory>();
}

std::shared_ptr<ArmBackend> createArmBackend() { return std::make_shared<ArmBackend>(); }

}  // namespace mllm::arm
