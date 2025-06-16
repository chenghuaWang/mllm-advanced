/**
 * @file KaiQuantizationPass.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Passes/KaiQuantizationPass.hpp"
#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Engine/CfgFile.hpp"
#include "mllm/IR/Passes/Pass.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

// Kernels
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

namespace mllm::arm {

bool KQP_linear_fp16_fp16_fp16p_mxk_kxn::match(const ir::op_ptr_t& op, const MllmModelCfg& cfg) {
  auto mllm_op = op->cast_<ir::linalg::LinearOp>()->getAOp();
  return cfg.opImplType(mllm_op->name()) == "KaiLinear_fp16_fp16_fp16p_mxk_kxn";
}

bool KQP_linear_fp16_fp16_fp16p_mxk_kxn::quantize(const ir::op_ptr_t& op, const MllmModelCfg& cfg) {
  auto mllm_op = (LinearOp*)(op->cast_<ir::linalg::LinearOp>()->getAOp());
  auto original_weight = mllm_op->weight();
  auto original_bias = mllm_op->bias();
  auto in_channels = mllm_op->cargo().in_channels;
  auto out_channels = mllm_op->cargo().out_channels;
  auto has_bias = mllm_op->cargo().bias;

  // FIXME:
  // If input is fp32, do type casting
  if (original_weight.dtype() != kFp16) {
    MLLM_WARN("Only support fp16 weight in KQP_linear_fp16_fp16_fp16p_mxk_kxn Pass");
    return false;
  }

  // FIXME:
  // If input is fp32, do type casting
  if (has_bias && (original_bias.dtype() != kFp16)) {
    MLLM_WARN("Only support fp16 bias in KQP_linear_fp16_fp16_fp16p_mxk_kxn Pass");
    return false;
  }

  auto weight_shape = original_weight.shape();

  // FIXME:
  // If input is not KxN Tensor, transpose it.
  if (weight_shape[0] != in_channels) {
    MLLM_WARN(
        "Only support in_channels x out_channels weight in KQP_linear_fp16_fp16_fp16p_mxk_kxn "
        "Pass, but found out_channels x in_channels weight");
    return false;
  }

  // pack_rhs_size return byte size.
  int32_t new_weights_size = kai_helper_.pack_rhs_size(in_channels, out_channels);

  // NOTE:
  // We used a flatter int8 buffer to represent the packed weight.
  // The packed weight can't be read or manipulated as a normal tensor.
  Tensor new_weights = Tensor::empty({new_weights_size}, kInt8, kCPU).alloc();

  // Perform quantize
  kai_helper_.pack_rhs_offline(new_weights.ptr<float16_t>(), original_weight.ptr<float16_t>(),
                               has_bias ? original_bias.ptr<float16_t>() : nullptr, in_channels,
                               out_channels);

  // Assign new weights to the linear op
  new_weights.setName(original_weight.name());
  mllm_op->weight() = new_weights;

  // Assign an one byte bias to occupy position.
  mllm_op->bias() = Tensor::empty({1}, kInt8, kCPU).alloc();

  return true;
}

std::shared_ptr<KQP_linear_fp16_fp16_fp16p_mxk_kxn> KQP_linear_fp16_fp16_fp16p_mxk_kxn::create() {
  return std::make_shared<KQP_linear_fp16_fp16_fp16p_mxk_kxn>();
}

bool KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk::match(const ir::op_ptr_t& op,
                                                    const MllmModelCfg& cfg) {
  auto mllm_op = op->cast_<ir::linalg::LinearOp>()->getAOp();

  std::string name = cfg.opImplType(mllm_op->name());
  std::vector<std::string> name_result;
  size_t name_prev = 0, name_pos;
  while ((name_pos = name.find(':', name_prev)) != std::string::npos) {
    name_result.push_back(name.substr(name_prev, name_pos - name_prev));
    name_prev = name_pos + 1;
  }
  if (name_prev < name.length()) { name_result.push_back(name.substr(name_prev)); }

  return name_result[0] == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk";
}

bool KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk::quantize(const ir::op_ptr_t& op,
                                                       const MllmModelCfg& cfg) {
  auto mllm_op = (LinearOp*)(op->cast_<ir::linalg::LinearOp>()->getAOp());

  std::string name = cfg.opImplType(mllm_op->name());
  std::vector<std::string> name_result;
  size_t name_prev = 0, name_pos;
  while ((name_pos = name.find(':', name_prev)) != std::string::npos) {
    name_result.push_back(name.substr(name_prev, name_pos - name_prev));
    name_prev = name_pos + 1;
  }
  if (name_prev < name.length()) { name_result.push_back(name.substr(name_prev)); }

  // Figure out which kernel to use
  // qai8dxp1x8_qsi4c32p4x8_1x4x32,
  // qai8dxp1x8_qsi4c32p8x8_1x8x32,
  // qai8dxp4x8_qsi4c32p4x8_8x4x32,
  // qai8dxp4x8_qsi4c32p4x8_16x4x32,
  // qai8dxp4x8_qsi4c32p8x8_4x8x32,
  // qai8dxp1x4_qsi4c32p4x4_1x4,
  KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg;
  if (name_result[1] == "qai8dxp1x8_qsi4c32p4x8_1x4x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32;
  } else if (name_result[1] == "qai8dxp1x8_qsi4c32p8x8_1x8x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;
  } else if (name_result[1] == "qai8dxp4x8_qsi4c32p4x8_8x4x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32;
  } else if (name_result[1] == "qai8dxp4x8_qsi4c32p4x8_16x4x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32;
  } else if (name_result[1] == "qai8dxp4x8_qsi4c32p8x8_4x8x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;
  } else if (name_result[1] == "qai8dxp1x4_qsi4c32p4x4_1x4") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4;
  }

  auto original_weight = mllm_op->weight();
  auto original_bias = mllm_op->bias();
  auto in_channels = mllm_op->cargo().in_channels;
  auto out_channels = mllm_op->cargo().out_channels;
  auto has_bias = mllm_op->cargo().bias;

  if (original_weight.dtype() != kFp32) {
    MLLM_WARN("Only support fp32 weight in KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk Pass");
    return false;
  }

  if (has_bias && (original_bias.dtype() != kFp32)) {
    MLLM_WARN("Only support fp32 bias in KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk Pass");
    return false;
  }

  auto weight_shape = original_weight.shape();

  // FIXME:
  // If input is not KxN Tensor, transpose it.
  if (weight_shape[0] != out_channels) {
    MLLM_WARN(
        "Only support out_channels x in_channels weight in KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk "
        "Pass, but found in_channels x out_channels weight");
    return false;
  }

  // pack_rhs_size return byte size.
  int32_t new_weights_size = kai_helper_.quant_pack_rhs_size(out_channels, in_channels, tile_cfg);

  // NOTE:
  // We used a flatter int8 buffer to represent the packed weight.
  // The packed weight can't be read or manipulated as a normal tensor.
  Tensor new_weights = Tensor::empty({new_weights_size}, kInt8, kCPU).alloc();

  // Perform quantize
  kai_helper_.quant_pack_rhs_offline(new_weights.ptr<uint8_t>(), original_weight.ptr<float>(),
                                     has_bias ? original_bias.ptr<float>() : nullptr, out_channels,
                                     in_channels, tile_cfg);

  // Assign new weights to the linear op
  new_weights.setName(original_weight.name());
  mllm_op->weight() = new_weights;

  // Assign an one byte bias to occupy position.
  mllm_op->bias() = Tensor::empty({1}, kInt8, kCPU).alloc();

  return true;
}

std::shared_ptr<KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk>
KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk::create() {
  return std::make_shared<KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk>();
}

bool KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn::match(const ir::op_ptr_t& op,
                                                    const MllmModelCfg& cfg) {
  auto mllm_op = op->cast_<ir::linalg::LinearOp>()->getAOp();

  std::string name = cfg.opImplType(mllm_op->name());
  std::vector<std::string> name_result;
  size_t name_prev = 0, name_pos;
  while ((name_pos = name.find(':', name_prev)) != std::string::npos) {
    name_result.push_back(name.substr(name_prev, name_pos - name_prev));
    name_prev = name_pos + 1;
  }
  if (name_prev < name.length()) { name_result.push_back(name.substr(name_prev)); }

  return name_result[0] == "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn";
}

bool KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn::quantize(const ir::op_ptr_t& op,
                                                       const MllmModelCfg& cfg) {
  auto mllm_op = (LinearOp*)(op->cast_<ir::linalg::LinearOp>()->getAOp());

  std::string name = cfg.opImplType(mllm_op->name());
  std::vector<std::string> name_result;
  size_t name_prev = 0, name_pos;
  while ((name_pos = name.find(':', name_prev)) != std::string::npos) {
    name_result.push_back(name.substr(name_prev, name_pos - name_prev));
    name_prev = name_pos + 1;
  }
  if (name_prev < name.length()) { name_result.push_back(name.substr(name_prev)); }

  // Figure out which kernel to use
  // qai8dxp1x8_qsi4c32p4x8_1x4x32,
  // qai8dxp1x8_qsi4c32p8x8_1x8x32,
  // qai8dxp4x8_qsi4c32p4x8_8x4x32,
  // qai8dxp4x8_qsi4c32p4x8_16x4x32,
  // qai8dxp4x8_qsi4c32p8x8_4x8x32,
  // qai8dxp1x4_qsi4c32p4x4_1x4,
  KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg;
  if (name_result[1] == "qai8dxp1x8_qsi4c32p4x8_1x4x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32;
  } else if (name_result[1] == "qai8dxp1x8_qsi4c32p8x8_1x8x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;
  } else if (name_result[1] == "qai8dxp4x8_qsi4c32p4x8_8x4x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32;
  } else if (name_result[1] == "qai8dxp4x8_qsi4c32p4x8_16x4x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32;
  } else if (name_result[1] == "qai8dxp4x8_qsi4c32p8x8_4x8x32") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;
  } else if (name_result[1] == "qai8dxp1x4_qsi4c32p4x4_1x4") {
    tile_cfg = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4;
  }

  auto original_weight = mllm_op->weight();
  auto original_bias = mllm_op->bias();
  auto in_channels = mllm_op->cargo().in_channels;
  auto out_channels = mllm_op->cargo().out_channels;
  auto has_bias = mllm_op->cargo().bias;

  if (original_weight.dtype() != kFp32) {
    MLLM_WARN("Only support fp32 weight in KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn Pass");
    return false;
  }

  if (has_bias && (original_bias.dtype() != kFp32)) {
    MLLM_WARN("Only support fp32 bias in KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn Pass");
    return false;
  }

  auto weight_shape = original_weight.shape();

  // FIXME:
  // If input is not KxN Tensor, transpose it.
  if (weight_shape[0] != in_channels) {
    MLLM_WARN(
        "Only support in_channels x out_channels weight in KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn "
        "Pass, but found out_channels x in_channels weight");
    return false;
  }

  // pack_rhs_size return byte size.
  int32_t new_weights_size = kai_helper_.quant_pack_rhs_size(in_channels, out_channels, tile_cfg);

  // NOTE:
  // We used a flatter int8 buffer to represent the packed weight.
  // The packed weight can't be read or manipulated as a normal tensor.
  Tensor new_weights = Tensor::empty({new_weights_size}, kInt8, kCPU).alloc();

  // Perform quantize
  kai_helper_.quant_pack_rhs_offline(new_weights.ptr<uint8_t>(), original_weight.ptr<float>(),
                                     has_bias ? original_bias.ptr<float>() : nullptr, in_channels,
                                     out_channels, tile_cfg);

  // Assign new weights to the linear op
  new_weights.setName(original_weight.name());
  mllm_op->weight() = new_weights;

  // Assign an one byte bias to occupy position.
  mllm_op->bias() = Tensor::empty({1}, kInt8, kCPU).alloc();

  return true;
}

std::shared_ptr<KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn>
KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn::create() {
  return std::make_shared<KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn>();
}

bool KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk::match(const ir::op_ptr_t& op,
                                                     const MllmModelCfg& cfg) {
  auto mllm_op = op->cast_<ir::linalg::LinearOp>()->getAOp();

  std::string name = cfg.opImplType(mllm_op->name());
  std::vector<std::string> name_result;
  size_t name_prev = 0, name_pos;
  while ((name_pos = name.find(':', name_prev)) != std::string::npos) {
    name_result.push_back(name.substr(name_prev, name_pos - name_prev));
    name_prev = name_pos + 1;
  }
  if (name_prev < name.length()) { name_result.push_back(name.substr(name_prev)); }

  return name_result[0] == "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk";
}

bool KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk::quantize(const ir::op_ptr_t& op,
                                                        const MllmModelCfg& cfg) {
  auto mllm_op = (LinearOp*)(op->cast_<ir::linalg::LinearOp>()->getAOp());

  std::string name = cfg.opImplType(mllm_op->name());
  std::vector<std::string> name_result;
  size_t name_prev = 0, name_pos;
  while ((name_pos = name.find(':', name_prev)) != std::string::npos) {
    name_result.push_back(name.substr(name_prev, name_pos - name_prev));
    name_prev = name_pos + 1;
  }
  if (name_prev < name.length()) { name_result.push_back(name.substr(name_prev)); }

  // Figure out which kernel to use
  // qsi8d32p1x8_qai4c32p4x8_1x4,
  // qsi8d32p4x4_qai4c32p4x4_8x4,
  // qsi8d32p4x8_qai4c32p4x8_8x4_i8mm,
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg;
  if (name_result[1] == "qsi8d32p1x8_qai4c32p4x8_1x4") {
    tile_cfg = KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::qsi8d32p1x8_qai4c32p4x8_1x4;
  } else if (name_result[1] == "qsi8d32p4x4_qai4c32p4x4_8x4") {
    tile_cfg = KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::qsi8d32p4x4_qai4c32p4x4_8x4;
  } else if (name_result[1] == "qsi8d32p4x8_qai4c32p4x8_8x4_i8mm") {
    tile_cfg = KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::qsi8d32p4x8_qai4c32p4x8_8x4_i8mm;
  }

  auto original_weight = mllm_op->weight();
  auto original_bias = mllm_op->bias();
  auto in_channels = mllm_op->cargo().in_channels;
  auto out_channels = mllm_op->cargo().out_channels;
  auto has_bias = mllm_op->cargo().bias;

  if (original_weight.dtype() != kFp32) {
    MLLM_WARN("Only support fp32 weight in KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk Pass");
    return false;
  }

  if (has_bias && (original_bias.dtype() != kFp32)) {
    MLLM_WARN("Only support fp32 bias in KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk Pass");
    return false;
  }

  auto weight_shape = original_weight.shape();

  // FIXME:
  // If input is not MxK Tensor, transpose it.
  if (weight_shape[1] != in_channels) {
    MLLM_WARN("Only support out_channels x in_channels weight in "
              "KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk "
              "Pass, but found in_channels x out_channels weight");
    return false;
  }

  // pack_rhs_size return byte size.
  int32_t new_weights_size = kai_helper_.quant_pack_rhs_size(in_channels, out_channels, tile_cfg);

  // NOTE:
  // We used a flatter int8 buffer to represent the packed weight.
  // The packed weight can't be read or manipulated as a normal tensor.
  Tensor new_weights = Tensor::empty({new_weights_size}, kInt8, kCPU).alloc();

  // Perform quantize
  kai_helper_.quant_pack_rhs_offline(new_weights.ptr<uint8_t>(), original_weight.ptr<float>(),
                                     has_bias ? original_bias.ptr<float>() : nullptr, in_channels,
                                     out_channels, tile_cfg);

  // Assign new weights to the linear op
  new_weights.setName(original_weight.name());
  mllm_op->weight() = new_weights;

  // Assign an one byte bias to occupy position.
  mllm_op->bias() = Tensor::empty({1}, kInt8, kCPU).alloc();

  return true;
}

std::shared_ptr<KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk>
KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk::create() {
  return std::make_shared<KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk>();
}

namespace MLLM_NAMESPACE_ANONYMOUS {

void visitCallGraph(KaiQuantizationPass* this_pass, const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op,
                    const MllmModelCfg& quant_cfg);

void visitSubGraph(KaiQuantizationPass* this_pass, const std::shared_ptr<ir::IRContext>& ir_ctx,
                   const ir::graph::SubGraphOp::self_ptr_t& subgraph_op,
                   const MllmModelCfg& quant_cfg) {
  auto region = subgraph_op->getTopRegion();
  for (auto& _op : region->ops()) {
    // If has call graph op
    if (_op->isa_<ir::graph::CallGraphOp>()) {
      visitCallGraph(this_pass, ir_ctx, _op->cast_<ir::graph::CallGraphOp>(), quant_cfg);
    } else if (_op->isa_<ir::linalg::LinearOp>()) {
      // KleidiAI Ops only optimized for linear.
      // Lookup quant_cfg and see if this linear should be quantized or not.
      if (!this_pass->performPatterns(_op)) {
        MLLM_WARN("Linear Op: {} is not quantized. Failed to match patterns or this Op is config "
                  "with no quantization.",
                  _op->cast_<ir::linalg::LinearOp>()->getAOp()->name());
      }
    }
    // FIXME:
    // We can also handle Conv2d / Conv3d or other Ops which can be represented as linear here.
  }
}

void visitCallGraph(KaiQuantizationPass* this_pass, const std::shared_ptr<ir::IRContext>& ir_ctx,
                    const ir::graph::CallGraphOp::self_ptr_t& call_op,
                    const MllmModelCfg& quant_cfg) {
  // Panic if input of call graph has no name
  auto& inputs = call_op->inputs();

  visitSubGraph(
      this_pass, ir_ctx,
      ir_ctx->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>(),
      quant_cfg);
}

}  // namespace MLLM_NAMESPACE_ANONYMOUS

KaiQuantizationPass::KaiQuantizationPass(const MllmModelCfg& cfg) : cfg_(cfg) {
  regPattern<KQP_linear_fp16_fp16_fp16p_mxk_kxn, KQP_linear_f32_qai8dxp_qsi4c32p_mxk_nxk,
             KQP_linear_f32_qai8dxp_qsi4c32p_mxk_kxn, KQP_linear_f16_qsi8d32p_qai4c32p_mxk_nxk>();
}

uint8_t KaiQuantizationPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  // Find the top CallGraphOp
  ir::graph::CallGraphOp::self_ptr_t call_main_graph_op = nullptr;
  r.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& remover,
          const ir::graph::CallGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
        // Make sure there is only one call graph op in the ModuleOp
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);

        call_main_graph_op = op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Visit all graph
  visitCallGraph(this, getCtx(), call_main_graph_op, cfg_);

  return ir::PASS_RET_SUCCESS;
}

bool KaiQuantizationPass::performPatterns(const ir::op_ptr_t& op) {
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

}  // namespace mllm::arm
