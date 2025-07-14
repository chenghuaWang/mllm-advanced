/**
 * @file CopyOpEliminatePass.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"
#include "mllm/Engine/CfgFile.hpp"

namespace mllm::ir {

class CopyOpEliminatePass final : public Pass {
 public:
  CopyOpEliminatePass() = default;

  ~CopyOpEliminatePass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

static inline std::shared_ptr<CopyOpEliminatePass> createCopyOpEliminatePass() {
  return std::make_shared<CopyOpEliminatePass>();
}

}  // namespace mllm::ir
