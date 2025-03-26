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

namespace mllm::arm {

ArmLinearOp::ArmLinearOp(const LinearOpCargo& cargo) : LinearOp(cargo) {}

void ArmLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto o = outputs[0];

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
