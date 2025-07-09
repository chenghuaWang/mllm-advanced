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
#include "mllm/Core/AOps/Conv3DOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/Arm/Kernels/sgemm.hpp"

// kai linear
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

namespace mllm::arm {

ArmConv3DOp::ArmConv3DOp(const Conv3DOpCargo& cargo) : Conv3DOp(cargo) {}

void ArmConv3DOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
  if (cargo_.bias) { bias_ = Tensor(ploader->operator[](name() + ".bias")); }
}

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

  if (activation.dtype() == kFp32 && cargo_.impl_type != Conv3DOpImplType::kDefault) {
    // transform the activation to im2col_activation
    Tensor im2col_activation = Tensor::empty({batch_size * out_time_size * out_h_size * out_w_size,
                                              cargo_.in_channels * cargo_.kernel_size[0]
                                                  * cargo_.kernel_size[1] * cargo_.kernel_size[2]},
                                             activation.dtype(), activation.device())
                                   .alloc();

    // activation after im2col:
    // [batch × time_out × h_out × w_out, in_channels × kernel_size_t ×
    // kernel_size_h × kernel_size_w]
    im2col_conv3d_p0_d1_activation_fp32(
        im2col_activation.ptr<float>(), activation.ptr<float>(), batch_size, cargo_.in_channels,
        in_time_size, in_h_size, in_w_size, cargo_.kernel_size[0], cargo_.kernel_size[1],
        cargo_.kernel_size[2], cargo_.stride[0], cargo_.stride[1], cargo_.stride[2]);

    // Weight after im2col is:
    // [out_channels, in_channels × kernel_size_t × kernel_size_h × kernel_size_w]

    // Bias after im2col is:
    // [out_channel]

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
      default: NYI("ArmConv3DOp's cargo_.impl_type is not supported yet."); break;
    }

    int permute_dims[] = {0, 4, 1, 2, 3};
    // View is shallow copy
    temp_output =
        temp_output.view({batch_size, out_time_size, out_h_size, out_w_size, cargo_.out_channels});

    // After permute, the final output shape is:
    // [B, O_channel, T, H, W]
    permute_fp32(temp_output.ptr<float>(), out.ptr<float>(), temp_output.shape().data(),
                 permute_dims, temp_output.shape().size());
    return;
  }

  if (activation.dtype() == kFp32 && cargo_.impl_type == Conv3DOpImplType::kDefault) {
    // Only support no bias version.
    MLLM_RT_ASSERT_EQ(cargo_.bias, false);

    auto batch_size = activation.shape()[0];
    auto in_channels = activation.shape()[1];
    auto out_channels = weight_.shape()[0];
    auto in_times = activation.shape()[2];
    auto in_height = activation.shape()[3];
    auto in_width = activation.shape()[4];

    auto out_times = outputs[0].shape()[2];
    auto out_height = outputs[0].shape()[3];
    auto out_width = outputs[0].shape()[4];

    auto kernel_t_size = weight_.shape()[2];
    auto kernel_h_size = weight_.shape()[3];
    auto kernel_w_size = weight_.shape()[4];

    auto a_ptr = activation.ptr<float>();
    auto w_ptr = weight_.ptr<float>();
    auto o_ptr = outputs[0].ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
      for (int ot = 0; ot < out_times; ++ot) {
        for (int oh = 0; oh < out_height; ++oh) {
          for (int ow = 0; ow < out_width; ++ow) {
            for (int oc = 0; oc < out_channels; ++oc) {
              float sum = 0.0f;
              for (int ic = 0; ic < in_channels; ++ic) {
                for (int kt = 0; kt < kernel_t_size; ++kt) {
                  for (int kh = 0; kh < kernel_h_size; ++kh) {
                    for (int kw = 0; kw < kernel_w_size; ++kw) {
                      int it = ot * kernel_t_size + kt;
                      int ih = oh * kernel_h_size + kh;
                      int iw = ow * kernel_w_size + kw;

                      int a_idx = b * (in_channels * in_times * in_height * in_width)
                                  + ic * (in_times * in_height * in_width)
                                  + it * (in_height * in_width) + ih * in_width + iw;

                      int w_idx = oc * (in_channels * kernel_t_size * kernel_h_size * kernel_w_size)
                                  + ic * (kernel_t_size * kernel_h_size * kernel_w_size)
                                  + kt * (kernel_h_size * kernel_w_size) + kh * kernel_w_size + kw;

                      sum += a_ptr[a_idx] * w_ptr[w_idx];
                    }
                  }
                }
              }
              int o_idx = b * (out_channels * out_times * out_height * out_width)
                          + oc * (out_times * out_height * out_width)
                          + ot * (out_height * out_width) + oh * out_width + ow;

              o_ptr[o_idx] = sum;
            }
          }
        }
      }
    }
  } else if (activation.dtype() == kFp16 && cargo_.impl_type == Conv3DOpImplType::kDefault) {
    MLLM_RT_ASSERT_EQ(cargo_.bias, false);

    auto batch_size = activation.shape()[0];
    auto in_channels = activation.shape()[1];
    auto out_channels = weight_.shape()[0];
    auto in_times = activation.shape()[2];
    auto in_height = activation.shape()[3];
    auto in_width = activation.shape()[4];

    auto out_times = outputs[0].shape()[2];
    auto out_height = outputs[0].shape()[3];
    auto out_width = outputs[0].shape()[4];

    auto kernel_t_size = weight_.shape()[2];
    auto kernel_h_size = weight_.shape()[3];
    auto kernel_w_size = weight_.shape()[4];

    auto a_ptr = activation.ptr<float16_t>();
    auto w_ptr = weight_.ptr<float16_t>();
    auto o_ptr = outputs[0].ptr<float16_t>();

    const int a_plane_size = in_times * in_height * in_width;
    const int w_plane_size = kernel_t_size * kernel_h_size * kernel_w_size;
    const int o_plane_size = out_times * out_height * out_width;

    for (int b = 0; b < batch_size; ++b) {
      for (int ot = 0; ot < out_times; ++ot) {
        for (int oh = 0; oh < out_height; ++oh) {
          for (int ow = 0; ow < out_width; ++ow) {
            for (int oc = 0; oc < out_channels; ++oc) {
              float sum = 0.0f;
              int ic = 0;
              for (; ic + 7 < in_channels; ic += 8) {
                float32x4_t sum_low = vdupq_n_f32(0.0f);
                float32x4_t sum_high = vdupq_n_f32(0.0f);

                for (int kt = 0; kt < kernel_t_size; ++kt) {
                  for (int kh = 0; kh < kernel_h_size; ++kh) {
                    for (int kw = 0; kw < kernel_w_size; ++kw) {
                      int it = ot * kernel_t_size + kt;
                      int ih = oh * kernel_h_size + kh;
                      int iw = ow * kernel_w_size + kw;

                      int a_idx_base = b * (in_channels * a_plane_size)
                                       + it * (in_height * in_width) + ih * in_width + iw;

                      int w_idx_base = oc * (in_channels * w_plane_size)
                                       + kt * (kernel_h_size * kernel_w_size) + kh * kernel_w_size
                                       + kw;

                      float16x8_t a_val = vld1q_f16(a_ptr + a_idx_base + ic * a_plane_size);

                      float16x8_t w_val = vld1q_f16(w_ptr + w_idx_base + ic * w_plane_size);

                      float32x4_t a_low = vcvt_f32_f16(vget_low_f16(a_val));
                      float32x4_t a_high = vcvt_f32_f16(vget_high_f16(a_val));
                      float32x4_t w_low = vcvt_f32_f16(vget_low_f16(w_val));
                      float32x4_t w_high = vcvt_f32_f16(vget_high_f16(w_val));

                      sum_low = vmlaq_f32(sum_low, a_low, w_low);
                      sum_high = vmlaq_f32(sum_high, a_high, w_high);
                    }
                  }
                }

                sum += vaddvq_f32(sum_low) + vaddvq_f32(sum_high);
              }

              for (; ic < in_channels; ++ic) {
                for (int kt = 0; kt < kernel_t_size; ++kt) {
                  for (int kh = 0; kh < kernel_h_size; ++kh) {
                    for (int kw = 0; kw < kernel_w_size; ++kw) {
                      int it = ot * kernel_t_size + kt;
                      int ih = oh * kernel_h_size + kh;
                      int iw = ow * kernel_w_size + kw;

                      int a_idx = b * (in_channels * a_plane_size) + ic * a_plane_size
                                  + it * (in_height * in_width) + ih * in_width + iw;

                      int w_idx = oc * (in_channels * w_plane_size) + ic * w_plane_size
                                  + kt * (kernel_h_size * kernel_w_size) + kh * kernel_w_size + kw;

                      sum += static_cast<float>(a_ptr[a_idx]) * static_cast<float>(w_ptr[w_idx]);
                    }
                  }
                }
              }
              int o_idx = b * (out_channels * o_plane_size) + oc * o_plane_size
                          + ot * (out_height * out_width) + oh * out_width + ow;

              o_ptr[o_idx] = sum;
            }
          }
        }
      }
    }
  }
}

}  // namespace mllm::arm