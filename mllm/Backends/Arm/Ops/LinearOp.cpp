/**
 * @file LinearOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/LinearOp.hpp"
#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Backends/Arm/Kernels/sgemm.hpp"
#include "mllm/Backends/Arm/Kernels/hgemm.hpp"

// kai linear
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

namespace mllm::arm {

ArmLinearOp::ArmLinearOp(const LinearOpCargo& cargo) : LinearOp(cargo) {}

void ArmLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto& o = outputs[0];

  switch (cargo_.impl_type_) {
    case LinearOpImplType::kKaiLinear_fp16_fp16_fp16p_mxk_kxn: {
      KaiLinear_fp16_fp16_fp16p_mxk_kxn kai_helper;
      kai_helper.matmul(o.ptr<float16_t>(), i.ptr<float16_t>(), weight_.ptr<float16_t>(),
                        i.shape()[i.shape().size() - 2], cargo_.in_channels, cargo_.out_channels);
      return;
    }
    case LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32: {
      auto M = i.shape()[i.shape().size() - 2];
      auto K = cargo_.in_channels;
      auto N = cargo_.out_channels;

      KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
      kai_helper.matmul(
          o.ptr<float>(), i.ptr<float>(), weight_.ptr<uint8_t>(), workspace.ptr<void>(), M, K, N,
          KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);
      return;
    }
    case LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32: {
      auto M = i.shape()[i.shape().size() - 2];
      auto K = cargo_.in_channels;
      auto N = cargo_.out_channels;

      KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
      kai_helper.matmul(
          o.ptr<float>(), i.ptr<float>(), weight_.ptr<uint8_t>(), workspace.ptr<void>(), M, K, N,
          KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32);
      return;
    }
    case LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32: {
      auto M = i.shape()[i.shape().size() - 2];
      auto K = cargo_.in_channels;
      auto N = cargo_.out_channels;

      KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
      kai_helper.matmul(
          o.ptr<float>(), i.ptr<float>(), weight_.ptr<uint8_t>(), workspace.ptr<void>(), M, K, N,
          KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32);
      return;
    }
    case LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32: {
      auto M = i.shape()[i.shape().size() - 2];
      auto K = cargo_.in_channels;
      auto N = cargo_.out_channels;

      KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
      kai_helper.matmul(
          o.ptr<float>(), i.ptr<float>(), weight_.ptr<uint8_t>(), workspace.ptr<void>(), M, K, N,
          KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32);
      return;
    }
    case LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32: {
      auto M = i.shape()[i.shape().size() - 2];
      auto K = cargo_.in_channels;
      auto N = cargo_.out_channels;

      KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
      kai_helper.matmul(
          o.ptr<float>(), i.ptr<float>(), weight_.ptr<uint8_t>(), workspace.ptr<void>(), M, K, N,
          KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32);
      return;
    }
    case LinearOpImplType::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4: {
      auto M = i.shape()[i.shape().size() - 2];
      auto K = cargo_.in_channels;
      auto N = cargo_.out_channels;

      KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();
      kai_helper.matmul(o.ptr<float>(), i.ptr<float>(), weight_.ptr<uint8_t>(),
                        workspace.ptr<void>(), M, K, N,
                        KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4);
      return;
    }
    default:
      // Fallback to mllm's self linear op impl.
      break;
  }

  // MxK @ NxK
  if (i.dtype() == DataTypes::kFp32 && o.dtype() == DataTypes::kFp32
      && weight_.dtype() == DataTypes::kFp32) {
    auto M = i.shape()[i.shape().size() - 2];
    auto K = cargo_.in_channels;
    auto N = cargo_.out_channels;
    size_t loops = 1;
    for (size_t idx = 0; idx < i.shape().size() - 2; ++idx) { loops *= i.shape()[idx]; }

    for (int l = 0; l < loops; ++l) {
      auto a_ptr = i.ptr<float>() + l * M * K;
      auto b_ptr = weight_.ptr<float>();
      auto c_ptr = o.ptr<float>() + l * M * N;
      auto bias_ptr = cargo_.bias ? bias_.ptr<float>() : nullptr;
      sgemm_mk_nk_mn_V1(a_ptr, b_ptr, c_ptr, M, K, N, bias_ptr, cargo_.thread());
    }

    return;
  }

  if (i.dtype() == DataTypes::kFp16 && o.dtype() == DataTypes::kFp16
      && weight_.dtype() == DataTypes::kFp16) {
    auto M = i.shape()[i.shape().size() - 2];
    auto K = cargo_.in_channels;
    auto N = cargo_.out_channels;
    size_t loops = 1;
    for (size_t idx = 0; idx < i.shape().size() - 2; ++idx) { loops *= i.shape()[idx]; }

    for (int l = 0; l < loops; ++l) {
      auto a_ptr = i.ptr<float16_t>() + l * M * K;
      auto b_ptr = weight_.ptr<float16_t>();
      auto c_ptr = o.ptr<float16_t>() + l * M * N;
      auto bias_ptr = cargo_.bias ? bias_.ptr<float16_t>() : nullptr;
      hgemm_mk_nk_mn_V1(a_ptr, b_ptr, c_ptr, M, K, N, bias_ptr, cargo_.thread());
    }

    return;
  }
}

}  // namespace mllm::arm
