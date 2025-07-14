/**
 * @file Conv3DOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/Conv3DOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

Conv3DOp::Conv3DOp(const Conv3DOpCargo& cargo) : BaseOp(OpType::kConv3D), cargo_(cargo) {}

void Conv3DOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
  if (cargo_.bias) { bias_ = Tensor(ploader->operator[](name() + ".bias")); }
}

void Conv3DOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::Conv3DOp>(shared_from_this(), i_irs, o_irs);
}

void Conv3DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("Conv3DOp::forward is not implemented");
}

void Conv3DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  const auto& ishape = i.shape();

  // Input must be 5D: [batch, channels, depth, height, width]
  if (ishape.size() != 5) {
    MLLM_ERROR_EXIT(kError, "Conv3DOp expects 5D input, got {} D", ishape.size());
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
    return;
  }

  // QNN May not support batch. You can write passes to eliminate batch dim.
  const int batch = ishape[0];
  const int in_channels = ishape[1];  // channel axis in VLM
  const int in_depth = ishape[2];     // time axis in VLM
  const int in_height = ishape[3];    // height axis in VLM
  const int in_width = ishape[4];     // width axis in VLM

  MLLM_RT_ASSERT_EQ(in_channels, cargo_.in_channels);

  // Retrieve convolution parameters from cargo_
  const auto& kernel = cargo_.kernel_size;  // [kd, kh, kw]
  const auto& stride = cargo_.stride;       // [sd, sh, sw]
  const int out_channels = cargo_.out_channels;

  // FIXME we not consider padding, dilation and group right now.
  // padding is always 0,
  // dilation is always 1,

  auto out_shape = [](int dim_size, int kernel_size, int stride_size, int padding_size,
                      int dilation_size) -> int32_t {
    // FIXME use floor.
    return ((dim_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size)
           + 1;
  };

  auto d_out = out_shape(in_depth, kernel[0], stride[0], 0, 1);
  auto h_out = out_shape(in_height, kernel[1], stride[1], 0, 1);
  auto w_out = out_shape(in_width, kernel[2], stride[2], 0, 1);

  auto new_shape = std::vector<int32_t>{batch, out_channels, d_out, h_out, w_out};

  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void Conv3DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

Conv3DOp::params_t Conv3DOp::params() {
  params_t ret;
  ret.insert({name() + ".weight", weight_});
  if (cargo_.bias) { ret.insert({name() + ".bias", bias_}); }
  return ret;
}

}  // namespace mllm
