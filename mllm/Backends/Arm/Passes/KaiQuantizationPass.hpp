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

#include <vector>

#include "mllm/Engine/CfgFile.hpp"
#include "mllm/IR/Passes/Pass.hpp"

// Kernels
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

namespace mllm::arm {

struct KaiQuantizationBasePattern {
  virtual bool match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) = 0;
  virtual bool quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) = 0;
};

struct KQP_linear_fp16_fp16_fp16p_mxk_kxn final : public KaiQuantizationBasePattern {
  bool match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;
  bool quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;

  KaiLinear_fp16_fp16_fp16p_mxk_kxn kai_helper_;
};

struct KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk final : public KaiQuantizationBasePattern {
  bool match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;
  bool quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;

  KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper_;
};

struct KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn final : public KaiQuantizationBasePattern {
  bool match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;
  bool quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) override;

  KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn kai_helper_;
};

class KaiQuantizationPass : public ir::Pass {
 public:
  explicit KaiQuantizationPass(const MllmModelCfg& cfg);

  ~KaiQuantizationPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

  template<typename... Args>
  void regPattern() {
    (..., (_reg_one_pattern<Args>()));
  }

  bool performPatterns(const ir::op_ptr_t& op);

 private:
  template<typename T>
  void _reg_one_pattern() {
    auto p = T::create();
    patterns_.emplace_back(p);
  }

  MllmModelCfg& cfg_;
  std::vector<std::shared_ptr<KaiQuantizationBasePattern>> patterns_;
};

static inline std::shared_ptr<KaiQuantizationPass> createKaiQuantizationPass(
    const MllmModelCfg& cfg) {
  return std::make_shared<KaiQuantizationPass>(cfg);
}

}  // namespace mllm::arm
