/**
 * @file QuantizePass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <regex>
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Passes/QuantizePass.hpp"
#include "mllm/IR/Tensor/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::ir {
bool CommonParamSimpleCastPattern::match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) {
  auto param_tensor = (op->outputs().front())->cast_<ir::tensor::TensorValue>();
  auto param_tensor_name = param_tensor->getSymbolAttr()->str();
  if (param_tensor_name.empty()) { return false; }
  for (auto& param_name : cfg.paramNames()) {
    std::regex re(param_name);
    if (std::regex_match(param_tensor_name, re)) { return true; }
  }
  return false;
}

bool CommonParamSimpleCastPattern::quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) {
  auto param_tensor = (op->outputs().front())->cast_<ir::tensor::TensorValue>();
  auto param_tensor_name = param_tensor->getSymbolAttr()->str();
  std::string key;
  for (auto& param_name : cfg.paramNames()) {
    std::regex re(param_name);
    if (std::regex_match(param_tensor_name, re)) { key = param_name; }
  }

  MLLM_INFO("Processing param: {}", param_tensor_name);

  auto dtype = cfg.paramDtype(key);

  auto mllm_tensor = param_tensor->tensor_;

  if (mllm_tensor.dtype() == dtype) { return true; }

  // Change the tensor this ir holds.
  param_tensor->tensor_ = mllm_tensor.to(dtype);

  return true;
}

std::shared_ptr<CommonParamSimpleCastPattern> CommonParamSimpleCastPattern::create() {
  return std::make_shared<CommonParamSimpleCastPattern>();
}

CommonParamQuantizePass::CommonParamQuantizePass(const MllmModelCfg& cfg) : cfg_(cfg) {
  regPattern<CommonParamSimpleCastPattern>();
}

uint8_t CommonParamQuantizePass::run(const ir::node_ptr_t& op) {
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find init_params graph
  auto init_params_graph =
      getCtx()->lookupSymbolTable("init_params")->cast_<ir::graph::SubGraphOp>();

  auto r = ir::IRWriter(getCtx(), init_params_graph->getTopRegion());
  r.walk<ir::tensor::AllocGlobalOp>(
      [&](ir::IRWriter& reader,
          const ir::tensor::AllocGlobalOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
        (void)performPatterns(op);
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  return ir::PASS_RET_SUCCESS;
}

bool CommonParamQuantizePass::performPatterns(const ir::op_ptr_t& op) {
  for (auto& p : patterns_) {
    if (p->match(op, cfg_)) {
      if (p->quantize(op, cfg_)) {
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

}  // namespace mllm::ir
