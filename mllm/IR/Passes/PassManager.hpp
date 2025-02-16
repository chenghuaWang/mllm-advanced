/**
 * @file PassManager.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <vector>
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::ir {

class PassManager {
 public:
  enum Pattern {
    GREEDY = 0,
  };

  PassManager() = delete;
  explicit PassManager(const std::shared_ptr<IRContext>& ctx);

  PassManager& reg(const pass_ptr_t& pass);

  void clear();

  bool run(Pattern p = GREEDY);

 private:
  std::shared_ptr<IRContext> ctx_;
  std::vector<pass_ptr_t> passes_;
};

}  // namespace mllm::ir