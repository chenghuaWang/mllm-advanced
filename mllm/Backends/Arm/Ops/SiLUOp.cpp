/**
 * @file SiLUOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/SiLUOp.hpp"
#include "mllm/Backends/Arm/Kernels/silu.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmSiLUOp::ArmSiLUOp(const SiLUOpCargo& cargo) : SiLUOp(cargo) {}

void ArmSiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto Y = outputs[0];

  switch (X.dtype()) {
    case kFp32: {
      silu_V1(X.ptr<float>(), Y.ptr<float>(), X.numel());
      break;
    }
    case kFp16: {
      silu_fp16_V1(X.ptr<float16_t>(), Y.ptr<float16_t>(), X.numel());
      break;
    }
    default: NYI("ArmSiLUOp::forward not support dtype {}", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm
