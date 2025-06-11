/**
 * @file KaiQuantizationPass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Engine/CfgFile.hpp"
#include "mllm/IR/Passes/Pass.hpp"

namespace mllm::arm {

class KaiQuantizationPass : public ir::Pass {
 public:
  explicit KaiQuantizationPass(const MllmModelCfg& cfg);

  ~KaiQuantizationPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

 private:
  MllmModelCfg& cfg_;
};

static inline std::shared_ptr<KaiQuantizationPass> createKaiQuantizationPass(
    const MllmModelCfg& cfg) {
  return std::make_shared<KaiQuantizationPass>(cfg);
}

}  // namespace mllm::arm
