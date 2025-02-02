/**
 * @file X86Backend.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/X86/X86Backend.hpp"
#include "mllm/Backends/X86/X86Allocator.hpp"
#include "mllm/Backends/X86/Ops/ElewiseOps.hpp"

namespace mllm::X86 {

X86Backend::X86Backend() : BackendBase(kCPU) {
  allocator_ = std::make_shared<X86Allocator>();
  regOpFactory<X86AddOpFactory, X86SubOpFactory, X86MulOpFactory, X86DivOpFactory>();
}

std::shared_ptr<X86Backend> createX86Backend() { return std::make_shared<X86Backend>(); }

}  // namespace mllm::X86
