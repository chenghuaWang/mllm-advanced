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
#include "mllm/Backends/Arm/Ops/Conv3DOp.hpp"
#include "mllm/Backends/Arm/Kernels/conv3d.hpp"
#include "mllm/Backends/Arm/Kernels/permute.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

// kai linear
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

namespace mllm::arm {

ArmConv3DOp::ArmConv3DOp(const Conv3DOpCargo& cargo) : Conv3DOp(cargo) {}

void ArmConv3DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& activation = inputs[0];
  auto& out = outputs[0];

  // Assume always has batch dim
  // [B, C, T, H, W]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 5);
  MLLM_RT_ASSERT_EQ(out.shape().size(), 5);

  auto batch_size = activation.shape()[0];
  auto in_time_size = activation.shape()[2];
  auto in_h_size = activation.shape()[3];
  auto in_w_size = activation.shape()[4];
  auto out_time_size = out.shape()[2];
  auto out_h_size = out.shape()[3];
  auto out_w_size = out.shape()[4];

  // transform the activation to im2col_activation
  Tensor im2col_activation = Tensor::empty({batch_size * out_time_size * out_h_size * out_w_size,
                                            cargo_.in_channels * cargo_.kernel_size[0]
                                                * cargo_.kernel_size[1] * cargo_.kernel_size[2]},
                                           activation.dtype(), activation.device())
                                 .alloc();

  if (activation.dtype() == kFp32) {
    // activation after im2col:
    // [batch × time_out × h_out × w_out, in_channels × kernel_size_t ×
    // kernel_size_h × kernel_size_w]
    im2col_conv3d_p0_d1_activation_fp32(im2col_activation.ptr<float>(), activation.ptr<float>(),
                                        batch_size, cargo_.in_channels, in_time_size, in_h_size,
                                        in_w_size, cargo_.kernel_size[0], cargo_.kernel_size[1],
                                        cargo_.kernel_size[2]);

    // Weight after im2col is:
    // [out_channels, in_channels × kernel_size_t × kernel_size_h × kernel_size_w]

    // Bias after im2col is:
    // TODO

    // Y after im2col is:
    // [batch × time_out × h_out × w_out, out_channels]

    // Temporary output
    Tensor temp_output =
        Tensor::empty({batch_size * out_time_size * out_h_size * out_w_size, cargo_.out_channels},
                      out.dtype(), out.device())
            .alloc();

    // We assume the weight and bias is already packing into kleidiai's tensor.
    switch (cargo_.impl_type) {
      case Conv3DOpImplType::
          kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32: {
        auto M = im2col_activation.shape()[0];
        auto K = im2col_activation.shape()[1];
        auto N = cargo_.out_channels;

        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;
        int32_t work_space_size = kai_helper.workspace_size(
            M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);
        auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
        kai_helper.matmul(
            temp_output.ptr<float>(), im2col_activation.ptr<float>(), weight_.ptr<uint8_t>(),
            workspace.ptr<void>(), M, K, N,
            KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);
        break;
      }
      case Conv3DOpImplType::
          kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32: {
        auto M = im2col_activation.shape()[0];
        auto K = im2col_activation.shape()[1];
        auto N = cargo_.out_channels;

        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;
        int32_t work_space_size = kai_helper.workspace_size(
            M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32);
        auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
        kai_helper.matmul(
            temp_output.ptr<float>(), im2col_activation.ptr<float>(), weight_.ptr<uint8_t>(),
            workspace.ptr<void>(), M, K, N,
            KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32);
        break;
      }
      case Conv3DOpImplType::
          kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32: {
        auto M = im2col_activation.shape()[0];
        auto K = im2col_activation.shape()[1];
        auto N = cargo_.out_channels;

        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;
        int32_t work_space_size = kai_helper.workspace_size(
            M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32);
        auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
        kai_helper.matmul(
            temp_output.ptr<float>(), im2col_activation.ptr<float>(), weight_.ptr<uint8_t>(),
            workspace.ptr<void>(), M, K, N,
            KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32);
        break;
      }
      case Conv3DOpImplType::
          kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32: {
        auto M = im2col_activation.shape()[0];
        auto K = im2col_activation.shape()[1];
        auto N = cargo_.out_channels;

        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;
        int32_t work_space_size = kai_helper.workspace_size(
            M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32);
        auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
        kai_helper.matmul(
            temp_output.ptr<float>(), im2col_activation.ptr<float>(), weight_.ptr<uint8_t>(),
            workspace.ptr<void>(), M, K, N,
            KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32);
        break;
      }
      case Conv3DOpImplType::
          kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32: {
        auto M = im2col_activation.shape()[0];
        auto K = im2col_activation.shape()[1];
        auto N = cargo_.out_channels;

        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;
        int32_t work_space_size = kai_helper.workspace_size(
            M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32);
        auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
        kai_helper.matmul(
            temp_output.ptr<float>(), im2col_activation.ptr<float>(), weight_.ptr<uint8_t>(),
            workspace.ptr<void>(), M, K, N,
            KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32);
        break;
      }
      case Conv3DOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4: {
        auto M = im2col_activation.shape()[0];
        auto K = im2col_activation.shape()[1];
        auto N = cargo_.out_channels;

        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;
        int32_t work_space_size = kai_helper.workspace_size(
            M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4);
        auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
        kai_helper.matmul(
            temp_output.ptr<float>(), im2col_activation.ptr<float>(), weight_.ptr<uint8_t>(),
            workspace.ptr<void>(), M, K, N,
            KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4);
        break;
      }
      default: NYI("ArmConv3DOp's cargo_.impl_type is not supported yet.");
    }

    int permute_dims[] = {0, 4, 1, 2, 3};
    // View is shallow copy
    temp_output.view({batch_size, out_time_size, out_h_size, out_w_size, cargo_.out_channels});

    // After permute, the final output shape is:
    // [B, O_channel, T, H, W]
    permute_fp32(temp_output.ptr<float>(), out.ptr<float>(), temp_output.shape().data(),
                 permute_dims, temp_output.shape().size());
  }
}

}  // namespace mllm::arm