/**
 * @file TransposeOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/TransposeOp.hpp"
#include <cstring>
#include "mllm/Backends/Arm/Kernels/transpose.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {
ArmTransposeOp::ArmTransposeOp(const TransposeOpCargo& cargo) : TransposeOp(cargo) {}

void ArmTransposeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto i = inputs[0];
  auto o = outputs[0];

  MLLM_RT_ASSERT(i.isContiguous());
  MLLM_RT_ASSERT(o.isContiguous());

  switch (i.dtype()) {
    case kFp32: {
      // [B, S, H, D] -> [B, H, S, D]
      if (i.shape().size() == 4 && o.shape().size() == 4
          && ((cargo_.transpose_dim_x == 1 && cargo_.transpose_dim_y == 2)
              || (cargo_.transpose_dim_x == 2 && cargo_.transpose_dim_y == 1))) {
        auto shape = i.shape();
        auto B = shape[0];
        auto S = shape[1];
        auto H = shape[2];
        auto D = shape[3];

        // if S == 1, there is no need to transpose.
        if (S == 1) {
          memcpy(o.ptr<float>(), i.ptr<float>(), B * S * H * D * sizeof(float));
          return;
        }

        transpose_bshd_bhsd(i.ptr<float>(), o.ptr<float>(), B, S, H, D);
        return;
      }
    }
    default:
      NYI("The dtype {} is not supported in transpose op yet", dataTypes2Str(i.dtype()));
      break;
  }

  NYI("The transpose on shape size {}, dim {} and {} is not supported yet", i.shape().size(),
      cargo_.transpose_dim_x, cargo_.transpose_dim_y);
}

void ArmTransposeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Common cases
  auto shape = inputs[0].shape();
  std::swap(shape[cargo_.transpose_dim_x], shape[cargo_.transpose_dim_y]);
  Tensor o = Tensor::empty(shape, inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(o);
}

void ArmTransposeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Common cases
  outputs[0].alloc();
}

}  // namespace mllm::arm
