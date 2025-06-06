/**
 * @file QnnAnonymousOpNamingPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::qnn {

class QnnAnonymousOpNamingPass : public ir::Pass {
 public:
  QnnAnonymousOpNamingPass() = default;
  ~QnnAnonymousOpNamingPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

static inline std::shared_ptr<QnnAnonymousOpNamingPass> createQnnAnonymousOpNamingPass() {
  return std::make_shared<QnnAnonymousOpNamingPass>();
}

}  // namespace mllm::qnn