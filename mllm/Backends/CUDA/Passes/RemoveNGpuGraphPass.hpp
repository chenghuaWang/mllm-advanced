/**
 * @file RemoveNGpuGraphPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"

using namespace mllm::ir;

namespace mllm::cuda {

class RemoveNonGpuGraphPass : public Pass {
 public:
  RemoveNonGpuGraphPass() = default;
  ~RemoveNonGpuGraphPass() override = default;

  uint8_t run(const node_ptr_t& op) override;
};

static inline std::shared_ptr<RemoveNonGpuGraphPass> createRemoveNonGpuGraphPass() {
  return std::make_shared<RemoveNonGpuGraphPass>();
}

}  // namespace mllm::cuda
