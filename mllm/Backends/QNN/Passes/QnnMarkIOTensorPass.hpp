/**
 * @file QnnMarkIOTensorPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::qnn {

class QnnMarkIOTensorPass : public ir::Pass {
 public:
  QnnMarkIOTensorPass() = default;
  ~QnnMarkIOTensorPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

static inline std::shared_ptr<QnnMarkIOTensorPass> createQnnMarkIOTensorPass() {
  return std::make_shared<QnnMarkIOTensorPass>();
}

}  // namespace mllm::qnn
