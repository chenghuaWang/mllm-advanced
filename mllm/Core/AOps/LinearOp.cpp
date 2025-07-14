/**
 * @file LinearOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

LinearOpImplType LinearOpCargo::parseLinearOpImplTypeStr(const std::string& type_str) {
  if (type_str == "Default") { return LinearOpImplType::kDefault; }
  if (type_str == "Kai_Start") { return LinearOpImplType::kKai_Start; }
  if (type_str == "KaiLinear_fp16_fp16_fp16p_mxk_kxn") {
    return LinearOpImplType::kKaiLinear_fp16_fp16_fp16p_mxk_kxn;
  }
  if (type_str == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp1x8_qsi4c32p4x8_1x4x32") {
    return LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32;
  }
  if (type_str == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp1x8_qsi4c32p8x8_1x8x32") {
    return LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  }
  if (type_str == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp4x8_qsi4c32p4x8_8x4x32") {
    return LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32;
  }
  if (type_str == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp4x8_qsi4c32p4x8_16x4x32") {
    return LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32;
  }
  if (type_str == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp4x8_qsi4c32p8x8_4x8x32") {
    return LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32;
  }
  if (type_str == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp1x4_qsi4c32p4x4_1x4") {
    return LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4;
  }
  if (type_str == "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk:qsi8d32p4x4_qai4c32p4x4_8x4") {
    return LinearOpImplType::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4;
  }
  if (type_str == "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk:qsi8d32p1x8_qai4c32p4x8_1x4") {
    return LinearOpImplType::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4;
  }
  if (type_str == "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk:qsi8d32p4x8_qai4c32p4x8_8x4_i8mm") {
    return LinearOpImplType::
        KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm;
  }
  if (type_str == "Kai_End") { return LinearOpImplType::kKai_End; }

  MLLM_WARN("Can't parse Linear Impl Type: {}. Fallback to default impl.", type_str);

  return LinearOpImplType::kDefault;
}

LinearOp::LinearOp(const LinearOpCargo& cargo) : BaseOp(OpType::kLinear), cargo_(cargo) {}

void LinearOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
  if (cargo_.bias) { bias_ = Tensor(ploader->operator[](name() + ".bias")); }
}

void LinearOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::LinearOp>(shared_from_this(), i_irs, o_irs);

  // Save parameters to global look up table
  for (auto& p : this->params()) {
    MLLM_RT_ASSERT_EQ(p.second.name(), p.first);
    auto v = ctx->create<ir::tensor::TensorValue>(p.second);
    v->name() = p.first;
    ctx->addToSymbolTable(v, p.first);
  }
}

void LinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LinearOp::forward is not implemented");
}

void LinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], cargo_.in_channels);

  auto o_shape = i_shape;
  o_shape[o_shape.size() - 1] = cargo_.out_channels;

  outputs.emplace_back(Tensor::empty(o_shape, i.dtype(), i.device()));
}

void LinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

LinearOp::params_t LinearOp::params() {
  params_t ret;
  ret.insert({name() + ".weight", weight_});
  if (cargo_.bias) { ret.insert({name() + ".bias", bias_}); }
  return ret;
}

}  // namespace mllm
