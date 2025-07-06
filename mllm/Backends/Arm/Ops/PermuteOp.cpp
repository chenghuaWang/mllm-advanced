/**
 * @file PermuteOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/PermuteOp.hpp"
#include "mllm/Backends/Arm/Kernels/permute.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmPermuteOp::ArmPermuteOp(const PermuteOpCargo& cargo) : PermuteOp(cargo) {}

void ArmPermuteOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  switch (X.dtype()) {
    case kFp32: {
      auto i_shape = X.shape();
      permute_fp32(X.ptr<float>(), Y.ptr<float>(), i_shape.data(), cargo_.permute_dims.data(),
                   i_shape.size());
      break;
    }
    case kFp16: {
      auto i_shape = X.shape();
      permute_fp16(X.ptr<float16_t>(), Y.ptr<float16_t>(), i_shape.data(),
                   cargo_.permute_dims.data(), i_shape.size());
      break;
    }
    default: NYI("ArmPermuteOp::forward not support dtype {}", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm