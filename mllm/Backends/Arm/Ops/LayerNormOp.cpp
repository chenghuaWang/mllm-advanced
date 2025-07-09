/**
 * @file LayerNormOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/LayerNormOp.hpp"
#include "mllm/Backends/Arm/Kernels/layernorm.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmLayerNormOp::ArmLayerNormOp(const LayerNormOpCargo& cargo) : LayerNormOp(cargo) {}

void ArmLayerNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  int32_t loop_size = 1;
  for (int i : cargo_.normalized_shape) { loop_size *= i; }

  MLLM_RT_ASSERT(loop_size > 0);
  MLLM_RT_ASSERT_EQ(X.numel() % loop_size, 0);

  // Calculate loop times
  size_t loop_times = X.numel() / loop_size;

  switch (X.dtype()) {
    case kFp32: {
      for (int l = 0; l < loop_times; ++l) {
        layernorm_N_fp32(Y.ptr<float>() + l * loop_size, X.ptr<float>() + l * loop_size,
                         cargo_.elementwise_affine ? weight_.ptr<float>() : nullptr,
                         cargo_.bias ? bias_.ptr<float>() : nullptr, loop_size, cargo_.eps);
      }
      break;
    }
    case kFp16: {
      for (int l = 0; l < loop_times; ++l) {
        layernorm_N_fp16(Y.ptr<float16_t>() + l * loop_size, X.ptr<float16_t>() + l * loop_size,
                         cargo_.elementwise_affine ? weight_.ptr<float16_t>() : nullptr,
                         cargo_.bias ? bias_.ptr<float16_t>() : nullptr, loop_size, cargo_.eps);
      }
      break;
    }
    default: NYI("ArmLayerNormOp::forward not support dtype {}", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm