/**
 * @file ElewiseOps.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/ElewiseOps.hpp"
#include "mllm/Backends/Arm/Kernels/element_wise.hpp"
#include <arm_fp16.h>

namespace mllm::arm {

void ArmAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& a_tensor = inputs[0];
  auto& b_tensor = inputs[1];
  auto& c_tensor = outputs[0];

  if (a_tensor.dtype() == kFp16 && b_tensor.dtype() == kFp16 && c_tensor.dtype() == kFp16) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_add_fp16(a_tensor.ptr<float16_t>(), b_tensor.ptr<float16_t>(), c_tensor.ptr<float16_t>(),
                  static_cast<int>(a_tensor.elementSize()));
      return;
    }
  }

  if (a_tensor.dtype() == kFp32 && b_tensor.dtype() == kFp32 && c_tensor.dtype() == kFp32) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_add_fp32(a_tensor.ptr<float>(), b_tensor.ptr<float>(), c_tensor.ptr<float>(),
                  static_cast<int>(a_tensor.elementSize()));
      return;
    }
  }
}

}  // namespace mllm::arm
