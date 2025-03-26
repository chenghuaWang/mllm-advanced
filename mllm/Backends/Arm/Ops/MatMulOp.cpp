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
#include "mllm/Backends/Arm/Kernels/hgemm.hpp"
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

    // fp32 @ fp32 -> fp32, all shape_len = 4
    if (i_a_shape.size() == 4 && i_b_shape.size() == 4 && i_a.dtype() == kFp32
        && i_b.dtype() == kFp32 && o.dtype() == kFp32) {
      auto DIM_0 = i_a_shape[0];
      auto DIM_1 = i_a_shape[1];

      // FIXME(LEVEL 0): Currently we use kleidiai kernel. And it's only support bias. so we need
      // to alloc bias. we should not use this.
      auto bias = new float[N];
      std::fill(bias, bias + N, 0);
      for (int d0 = 0; d0 < DIM_0; d0++) {
        for (int d1 = 0; d1 < DIM_1; d1++) {
          sgemm_mk_kn_mn_V1(i_a.offsettedPtr<float>({d0, d1, 0, 0}),
                            i_b.offsettedPtr<float>({d0, d1, 0, 0}),
                            o.offsettedPtr<float>({d0, d1, 0, 0}), M, K, N, bias, cargo_.thread());
        }
      }
      delete[] bias;
      return;
    }

    // fp32 @ fp32 -> fp32, all shape_len != 4
    if (i_a_shape.size() != 4 && i_b_shape.size() != 4 && i_a.dtype() == kFp32
        && i_b.dtype() == kFp32 && o.dtype() == kFp32) {
      auto A = i_a.ptr<float>();
      auto B = i_b.ptr<float>();
      auto C = o.ptr<float>();
      size_t loops = 1;
      for (int i = 0; i < i_a_shape.size() - 2; i++) { loops *= i_a_shape[i]; }

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

    // fp16 @ fp16 -> fp16, all shape_len = 4
    if (i_a_shape.size() == 4 && i_b_shape.size() == 4 && i_a.dtype() == kFp16
        && i_b.dtype() == kFp16 && o.dtype() == kFp16) {
      auto DIM_0 = i_a_shape[0];
      auto DIM_1 = i_a_shape[1];

      // FIXME(LEVEL 0): Currently we use kleidiai kernel. And it's only support bias. so we need
      // to alloc bias. we should not use this.
      auto bias = new float16_t[N];
      std::fill(bias, bias + N, 0);
      for (int d0 = 0; d0 < DIM_0; d0++) {
        for (int d1 = 0; d1 < DIM_1; d1++) {
          hgemm_mk_kn_mn_V1(i_a.offsettedPtr<float16_t>({d0, d1, 0, 0}),
                            i_b.offsettedPtr<float16_t>({d0, d1, 0, 0}),
                            o.offsettedPtr<float16_t>({d0, d1, 0, 0}), M, K, N, bias,
                            cargo_.thread());
        }
      }
      delete[] bias;
      return;
    }

    // fp16 @ fp16 -> fp16, all shape_len != 4
    if (i_a_shape.size() != 4 && i_b_shape.size() != 4 && i_a.dtype() == kFp16
        && i_b.dtype() == kFp16 && o.dtype() == kFp16) {
      auto A = i_a.ptr<float16_t>();
      auto B = i_b.ptr<float16_t>();
      auto C = o.ptr<float16_t>();
      size_t loops = 1;
      for (int i = 0; i < i_a_shape.size() - 2; i++) { loops *= i_a_shape[i]; }

      for (size_t l = 0; l < loops; l++) {
        auto a_ptr = A + l * M * K;
        auto b_ptr = broad_cast_flag ? B : B + l * M * K;
        auto c_ptr = C + l * M * N;

        // FIXME(LEVEL 0): Currently we use kleidiai kernel. And it's only support bias. so we need
        // to alloc bias. we should not use this.
        auto bias = new float16_t[N];
        std::fill(bias, bias + N, 0);
        hgemm_mk_kn_mn_V1(a_ptr, b_ptr, c_ptr, M, K, N, bias, cargo_.thread());
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

    // fp32 @ fp32 -> fp32
    if (i_a_shape.size() == 4 && i_b_shape.size() == 4 && i_a.dtype() == kFp32
        && i_b.dtype() == kFp32 && o.dtype() == kFp32) {
      auto DIM_0 = i_a_shape[0];
      auto DIM_1 = i_a_shape[1];
      for (int d0 = 0; d0 < DIM_0; d0++) {
        for (int d1 = 0; d1 < DIM_1; d1++) {
          sgemm_mk_nk_mn_V1(
              i_a.offsettedPtr<float>({d0, d1, 0, 0}), i_b.offsettedPtr<float>({d0, d1, 0, 0}),
              o.offsettedPtr<float>({d0, d1, 0, 0}), M, K, N, nullptr, cargo_.thread());
        }
      }
      return;
    }

    // fp32 @ fp32 -> fp32
    if (i_a_shape.size() != 4 && i_b_shape.size() != 4 && i_a.dtype() == kFp32
        && i_b.dtype() == kFp32 && o.dtype() == kFp32) {
      size_t loops = 1;
      for (int i = 0; i < i_a_shape.size() - 2; i++) { loops *= i_a_shape[i]; }

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

    // fp16 @ fp16 -> fp16
    if (i_a_shape.size() == 4 && i_b_shape.size() == 4 && i_a.dtype() == kFp16
        && i_b.dtype() == kFp16 && o.dtype() == kFp16) {
      auto DIM_0 = i_a_shape[0];
      auto DIM_1 = i_a_shape[1];
      for (int d0 = 0; d0 < DIM_0; d0++) {
        for (int d1 = 0; d1 < DIM_1; d1++) {
          hgemm_mk_nk_mn_V1(i_a.offsettedPtr<float16_t>({d0, d1, 0, 0}),
                            i_b.offsettedPtr<float16_t>({d0, d1, 0, 0}),
                            o.offsettedPtr<float16_t>({d0, d1, 0, 0}), M, K, N, nullptr,
                            cargo_.thread());
        }
      }
      return;
    }

    // fp16 @ fp16 -> fp16
    if (i_a_shape.size() != 4 && i_b_shape.size() != 4 && i_a.dtype() == kFp16
        && i_b.dtype() == kFp16 && o.dtype() == kFp16) {
      size_t loops = 1;
      for (int i = 0; i < i_a_shape.size() - 2; i++) { loops *= i_a_shape[i]; }

      auto A = i_a.ptr<float16_t>();
      auto B = i_b.ptr<float16_t>();
      auto C = o.ptr<float16_t>();
      for (size_t l = 0; l < loops; l++) {
        auto a_ptr = A + l * M * K;
        auto b_ptr = broad_cast_flag ? B : B + l * M * K;
        auto c_ptr = C + l * M * N;
        hgemm_mk_nk_mn_V1(a_ptr, b_ptr, c_ptr, M, K, N, nullptr, cargo_.thread());
      }
      return;
    }
  }
}

}  // namespace mllm::arm
