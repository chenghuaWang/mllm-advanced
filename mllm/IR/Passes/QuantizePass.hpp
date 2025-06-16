/**
 * @file QuantizePass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/IR/Passes/Pass.hpp"
#include "mllm/Engine/CfgFile.hpp"

namespace mllm::ir {

struct CommonParamQuantizePattern {
  virtual bool match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) = 0;
  virtual bool quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) = 0;
};

struct CommonParamSimpleCastPattern : public CommonParamQuantizePattern {
  bool match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;
  bool quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;

  static std::shared_ptr<CommonParamSimpleCastPattern> create();
};

class CommonParamQuantizePass final : public Pass {
 public:
  explicit CommonParamQuantizePass(const MllmModelCfg& cfg);

  ~CommonParamQuantizePass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

 private:
  template<typename... Args>
  void regPattern() {
    (..., (_reg_one_pattern<Args>()));
  }

  bool performPatterns(const ir::op_ptr_t& op);

  template<typename T>
  void _reg_one_pattern() {
    auto p = T::create();
    patterns_.emplace_back(p);
  }

  const MllmModelCfg& cfg_;
  std::vector<std::shared_ptr<CommonParamQuantizePattern>> patterns_;
};

static inline std::shared_ptr<CommonParamQuantizePass> createCommonParamQuantizePass(
    const MllmModelCfg& cfg) {
  return std::make_shared<CommonParamQuantizePass>(cfg);
}

}  // namespace mllm::ir