/**
 * @file SoftmaxOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/SoftmaxOp.hpp"
#include "mllm/Backends/Arm/Kernels/softmax.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmSoftmaxOp::ArmSoftmaxOp(const SoftmaxOpCargo& cargo) : SoftmaxOp(cargo) {}

void ArmSoftmaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto X = inputs[0];
  auto Y = outputs[0];
  MLLM_RT_ASSERT_EQ(X.shape().size(), 4);
  MLLM_RT_ASSERT_EQ(cargo_.axis, -1);
  auto B = X.shape()[0];
  auto H = X.shape()[1];
  auto S = X.shape()[2];
  auto D = X.shape()[3];
  switch (X.dtype()) {
    case kFp32:
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            softmax_V1(X.offsettedPtr<float>({b, h, s, 0}), Y.offsettedPtr<float>({b, h, s, 0}), D,
                       1);
          }
        }
      }
      break;
    case kFp16:
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            hsoftmax_V1(X.offsettedPtr<float16_t>({b, h, s, 0}),
                        Y.offsettedPtr<float16_t>({b, h, s, 0}), D, 1);
          }
        }
      }
      break;
    default: NYI("ArmSoftmaxOp::forward not support dtype {}", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm
