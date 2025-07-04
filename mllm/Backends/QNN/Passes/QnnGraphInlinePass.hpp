/**
 * @file QnnGraphInlinePass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::qnn {

class QnnGraphInlinePass : public ir::Pass {
 public:
  QnnGraphInlinePass() = default;
  ~QnnGraphInlinePass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

  std::vector<std::string> graph_need_to_be_inlined_;
};

static inline std::shared_ptr<QnnGraphInlinePass> createQnnGraphInlinePass(
    const std::vector<std::string>& graphs_need_to_be_inlined) {
  auto ret = std::make_shared<QnnGraphInlinePass>();
  ret->graph_need_to_be_inlined_ = graphs_need_to_be_inlined;
  return ret;
}

}  // namespace mllm::qnn
