/**
 * @file LinearOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

enum class LinearOpImplType {
  kDefault = 0,
  kKai_Start = 1,
  kKaiLinear_fp16_fp16_fp16p_mxk_kxn = 2,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32 = 3,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32 = 4,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32 = 5,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32 = 6,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32 = 7,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4 = 8,

  // left 9->255 for other kleidiai impl of linear

  kKai_End = 256,
};

// When creating layers/ops. It's recommend to use OpCargo if you want to specific the
// inputs/outputs' dtype.
//
// auto x =
//     LinearOpCargo{
//         .in_channels = 1024,
//         .out_channels = 1024,
//         .bias = true,
//         .transpose = false,
//     }
//         .setInputsDtype(LinearOpCargo::InputsPos::kInput, kInt8)
//         .setInputsDtype(LinearOpCargo::InputsPos::kWeight, kInt8)
//         .setOutputsDtype(LinearOpCargo::OutputsPos::kOutput, kInt8);
struct LinearOpCargo : public BaseOpCargo<LinearOpCargo> {
  enum InputsPos : size_t {  // NOLINT
    kInput = 0,
    kWeight = 1,
    kBias = 2,
  };

  enum OutputsPos : size_t {  // NOLINT
    kOutput = 0,
  };

  int32_t in_channels = 0;
  int32_t out_channels = 0;
  bool bias = true;
  bool transpose = false;

  // linear impl type.
  LinearOpImplType impl_type_ = LinearOpImplType::kDefault;
};

class LinearOp : public BaseOp {
 public:
  explicit LinearOp(const LinearOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  params_t params() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline const LinearOpCargo& cargo() const { return cargo_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  LinearOpCargo cargo_;
};

}  // namespace mllm
