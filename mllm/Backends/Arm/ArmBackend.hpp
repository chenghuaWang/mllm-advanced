/**
 * @file ArmBackend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Engine/BackendBase.hpp"

#ifndef __ARM_NEON
#error "Mllm's Arm backend only support those devices that have neon support"
#endif

#if __ARM_ARCH < 8
#error "Mllm's Arm backend only support those devices that have armv8 or above"
#endif

namespace mllm::arm {

class ArmBackend final : public BackendBase {
 public:
  ArmBackend();
};

std::shared_ptr<ArmBackend> createArmBackend();

}  // namespace mllm::arm
