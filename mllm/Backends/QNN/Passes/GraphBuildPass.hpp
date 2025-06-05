/**
 * @file GraphBuildPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <vector>
#include <string>
#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::qnn {

class GraphBuildPass : public ir::Pass {
 public:
  GraphBuildPass() = default;
  ~GraphBuildPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

  std::vector<std::string> graph_need_to_be_compiled_;
};

static inline std::shared_ptr<GraphBuildPass> createGraphBuildPass(
    const std::vector<std::string>& graphs_need_to_be_compiled) {
  auto ret = std::make_shared<GraphBuildPass>();
  ret->graph_need_to_be_compiled_ = graphs_need_to_be_compiled;
  return ret;
}

}  // namespace mllm::qnn
