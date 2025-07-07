/**
 * @file Conv3DOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {
enum class Conv3DOpImplType {
  kDefault = 0,
  kKai_Start = 1,
  kKaiLinear_fp16_fp16_fp16p_mxk_kxn = 2,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32 = 3,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32 = 4,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32 = 5,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32 = 6,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32 = 7,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4 = 8,

  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4 = 9,
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4 = 10,
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm = 11,

  // left 9->255 for other kleidiai impl of linear

  kKai_End = 256,
};

struct Conv3DOpCargo : public BaseOpCargo<Conv3DOpCargo> {
  int32_t in_channels;
  int32_t out_channels;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool bias = true;
  Conv3DOpImplType impl_type = Conv3DOpImplType::kDefault;
};

class Conv3DOp : public BaseOp {
 public:
  explicit Conv3DOp(const Conv3DOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  params_t params() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline const Conv3DOpCargo& cargo() const { return cargo_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  Conv3DOpCargo cargo_;
};

}  // namespace mllm
