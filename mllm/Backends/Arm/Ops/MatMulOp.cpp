/**
 * @file MatMulOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/MatMulOp.hpp"
#include "mllm/Backends/Arm/Kernels/sgemm.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm::arm {

ArmMatMulOp::ArmMatMulOp(const MatMulOpCargo& cargo) : MatMulOp(cargo) {}

void ArmMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto i_a = inputs[0];
  auto i_b = inputs[1];
  auto o = outputs[0];
  auto i_a_shape = i_a.shape();
  auto i_b_shape = i_b.shape();

  bool broad_cast_flag = i_a_shape.size() != i_b_shape.size();

  if (broad_cast_flag && i_b_shape.size() != 2) {
    MLLM_ERROR_EXIT(kError, "ArmMatMulOp's broadcasting only support 2D RHS matrix.");
  }

  // MxK @ KxN
  if (!cargo_.transpose_a && !cargo_.transpose_b) {
    auto M = i_a_shape[i_a_shape.size() - 2];
    auto K = i_a_shape[i_a_shape.size() - 1];
    auto N = i_b_shape[i_b_shape.size() - 1];
    size_t loops = 1;
    for (int i = 0; i < i_a_shape.size() - 2; i++) { loops *= i_a_shape[i]; }

    // fp32 @ fp32 -> fp32
    if (i_a.dtype() == kFp32 && i_b.dtype() == kFp32 && o.dtype() == kFp32) {
      auto A = i_a.ptr<float>();
      auto B = i_b.ptr<float>();
      auto C = o.ptr<float>();
      for (size_t l = 0; l < loops; l++) {
        auto a_ptr = A + l * M * K;
        auto b_ptr = broad_cast_flag ? B : B + l * M * K;
        auto c_ptr = C + l * M * N;

        // FIXME(LEVEL 0): Currently we use kleidiai kernel. And it's only support bias. so we need
        // to alloc bias. we should not use this.
        auto bias = new float[N];
        std::fill(bias, bias + N, 0);
        sgemm_mk_kn_mn_V1(a_ptr, b_ptr, c_ptr, M, K, N, bias, cargo_.thread());
        delete[] bias;
      }
      return;
    }
  }

  // MxK @ NxK
  if (!cargo_.transpose_a && cargo_.transpose_b) {
    auto M = i_a_shape[i_a_shape.size() - 2];
    auto K = i_a_shape[i_a_shape.size() - 1];
    auto N = i_b_shape[i_b_shape.size() - 2];
    size_t loops = 1;
    for (int i = 0; i < i_a_shape.size() - 2; i++) { loops *= i_a_shape[i]; }

    // fp32 @ fp32 -> fp32
    if (i_a.dtype() == kFp32 && i_b.dtype() == kFp32 && o.dtype() == kFp32) {
      auto A = i_a.ptr<float>();
      auto B = i_b.ptr<float>();
      auto C = o.ptr<float>();
      for (size_t l = 0; l < loops; l++) {
        auto a_ptr = A + l * M * K;
        auto b_ptr = broad_cast_flag ? B : B + l * M * K;
        auto c_ptr = C + l * M * N;
        sgemm_mk_nk_mn_V1(a_ptr, b_ptr, c_ptr, M, K, N, nullptr, cargo_.thread());
      }
      return;
    }
  }
}

}  // namespace mllm::arm
