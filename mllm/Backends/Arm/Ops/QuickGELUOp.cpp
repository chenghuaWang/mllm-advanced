/**
 * @file QuickGELUOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/QuickGELUOp.hpp"
#include "mllm/Backends/Arm/Kernels/gelu.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmQuickGELUOp::ArmQuickGELUOp(const QuickGELUOpCargo& cargo) : QuickGELUOp(cargo) {}

void ArmQuickGELUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  switch (X.dtype()) {
    case kFp32: {
      quick_gelu_fp32(Y.ptr<float>(), X.ptr<float>(), X.numel());
      break;
    }
    case kFp16: {
      quick_gelu_fp16(Y.ptr<float16_t>(), X.ptr<float16_t>(), X.numel());
      break;
    }
    default: NYI("ArmQuickGELUOp::forward not support dtype {}", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm
