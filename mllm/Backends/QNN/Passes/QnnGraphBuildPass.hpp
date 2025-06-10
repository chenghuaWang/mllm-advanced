/**
 * @file QnnGraphBuildPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::qnn {

class QnnGraphBuildPass : public ir::Pass {
 public:
  QnnGraphBuildPass();

  ~QnnGraphBuildPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

  std::vector<std::string> graph_need_to_be_compiled_;

  template<typename... Args>
  void regPattern() {
    (..., (_reg_one_pattern<Args>()));
  }

 private:
  template<typename T>
  void _reg_one_pattern() {
    auto pair = T::create();
    patterns_.insert({pair.first, pair.second});
  }

  void buildQnnLego(const ir::graph::SubGraphOp::self_ptr_t& sub_graph_op);

  std::unordered_map<OpType, std::shared_ptr<QnnBaseOpPattern>> patterns_;
};

static inline std::shared_ptr<QnnGraphBuildPass> createQnnGraphBuildPass(
    const std::vector<std::string>& graphs_need_to_be_compiled) {
  auto ret = std::make_shared<QnnGraphBuildPass>();
  ret->graph_need_to_be_compiled_ = graphs_need_to_be_compiled;
  return ret;
}

}  // namespace mllm::qnn
