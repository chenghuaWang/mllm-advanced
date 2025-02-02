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

namespace mllm::X86 {

class X86Backend final : public BackendBase {
 public:
  X86Backend();
};

std::shared_ptr<X86Backend> createX86Backend();

}  // namespace mllm::X86
