/**
 * @file TensorNamingPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::qnn {

class QnnTensorNamingPass : public ir::Pass {
 public:
  QnnTensorNamingPass() = default;
  ~QnnTensorNamingPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

static inline std::shared_ptr<QnnTensorNamingPass> createQnnTensorNamingPass() {
  return std::make_shared<QnnTensorNamingPass>();
}

}  // namespace mllm::qnn
