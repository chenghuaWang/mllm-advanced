/**
 * @file FillOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/FillOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Utils/Log.hpp"
#include <arm_neon.h>

namespace mllm {

ArmFillOp::ArmFillOp(const FillOpCargo& cargo) : FillOp(cargo) {}

void ArmFillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
  auto& t = inputs[0];
  auto dtype = t.dtype();
  // type
  // 0 -> zeros
  // 1 -> ones
  // 2 -> specific
  // 3 -> random
  // 4 -> arrange
  switch (cargo_.type) {
    case 0: {
      switch (dtype) {
        case kFp32: std::memset(t.ptr<float>(), 0, t.numel() * sizeof(float)); break;
        case kFp16:
          std::fill(t.ptr<float16_t>(), t.ptr<float16_t>() + t.numel(),
                    static_cast<float16_t>(0.f));
          break;
        default: NYI("ArmFillOp type=0, dtype={}.", dataTypes2Str(dtype));
      }
      break;
    }
    case 1: {
      switch (dtype) {
        case kFp32: std::fill(t.ptr<float>(), t.ptr<float>() + t.numel(), 1.f); break;
        case kFp16:
          std::fill(t.ptr<float16_t>(), t.ptr<float16_t>() + t.numel(),
                    static_cast<float16_t>(1.f));
          break;
        default: NYI("ArmFillOp type=0, dtype={}.", dataTypes2Str(dtype));
      }
      break;
    }
    case 2:
    case 3:
    case 4:
    default:
      MLLM_WARN("ArmFillOp found cargo.type={}, which is not supported yet. The ArmFillOp will do "
                "nothing on input tensor.",
                cargo_.type);
      break;
  }
}

}  // namespace mllm
