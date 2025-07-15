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
#include "mllm/Utils/Common.hpp"
#include <arm_fp16.h>

namespace mllm::arm {

void ArmAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& a_tensor = inputs[0];
  auto& b_tensor = inputs[1];
  auto& c_tensor = outputs[0];

  if (a_tensor.dtype() == kFp16 && b_tensor.dtype() == kFp16 && c_tensor.dtype() == kFp16) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_add_fp16(a_tensor.ptr<float16_t>(), b_tensor.ptr<float16_t>(), c_tensor.ptr<float16_t>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a + 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_add_constant_fp16(a_tensor.ptr<float16_t>(), *(b_tensor.ptr<float16_t>()),
                           c_tensor.ptr<float16_t>(), static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  if (a_tensor.dtype() == kFp32 && b_tensor.dtype() == kFp32 && c_tensor.dtype() == kFp32) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_add_fp32(a_tensor.ptr<float>(), b_tensor.ptr<float>(), c_tensor.ptr<float>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a + 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_add_constant_fp32(a_tensor.ptr<float>(), *(b_tensor.ptr<float>()), c_tensor.ptr<float>(),
                           static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  NYI("ArmAddOp::forward op not support current inputs");
}

void ArmSubOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& a_tensor = inputs[0];
  auto& b_tensor = inputs[1];
  auto& c_tensor = outputs[0];

  if (a_tensor.dtype() == kFp16 && b_tensor.dtype() == kFp16 && c_tensor.dtype() == kFp16) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_sub_fp16(a_tensor.ptr<float16_t>(), b_tensor.ptr<float16_t>(), c_tensor.ptr<float16_t>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a - 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_sub_constant_fp16(a_tensor.ptr<float16_t>(), *(b_tensor.ptr<float16_t>()),
                           c_tensor.ptr<float16_t>(), static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  if (a_tensor.dtype() == kFp32 && b_tensor.dtype() == kFp32 && c_tensor.dtype() == kFp32) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_sub_fp32(a_tensor.ptr<float>(), b_tensor.ptr<float>(), c_tensor.ptr<float>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a - 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_sub_constant_fp32(a_tensor.ptr<float>(), *(b_tensor.ptr<float>()), c_tensor.ptr<float>(),
                           static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  NYI("ArmSubOp::forward op not support current inputs");
}

void ArmMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& a_tensor = inputs[0];
  auto& b_tensor = inputs[1];
  auto& c_tensor = outputs[0];

  if (a_tensor.dtype() == kFp16 && b_tensor.dtype() == kFp16 && c_tensor.dtype() == kFp16) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_mul_fp16(a_tensor.ptr<float16_t>(), b_tensor.ptr<float16_t>(), c_tensor.ptr<float16_t>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a * 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_mul_constant_fp16(a_tensor.ptr<float16_t>(), *(b_tensor.ptr<float16_t>()),
                           c_tensor.ptr<float16_t>(), static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  if (a_tensor.dtype() == kFp32 && b_tensor.dtype() == kFp32 && c_tensor.dtype() == kFp32) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_mul_fp32(a_tensor.ptr<float>(), b_tensor.ptr<float>(), c_tensor.ptr<float>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a * 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_mul_constant_fp32(a_tensor.ptr<float>(), *(b_tensor.ptr<float>()), c_tensor.ptr<float>(),
                           static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  NYI("ArmMulOp::forward op not support current inputs");
}

void ArmDivOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& a_tensor = inputs[0];
  auto& b_tensor = inputs[1];
  auto& c_tensor = outputs[0];

  if (a_tensor.dtype() == kFp16 && b_tensor.dtype() == kFp16 && c_tensor.dtype() == kFp16) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_div_fp16(a_tensor.ptr<float16_t>(), b_tensor.ptr<float16_t>(), c_tensor.ptr<float16_t>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a * 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_div_constant_fp16(a_tensor.ptr<float16_t>(), *(b_tensor.ptr<float16_t>()),
                           c_tensor.ptr<float16_t>(), static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  if (a_tensor.dtype() == kFp32 && b_tensor.dtype() == kFp32 && c_tensor.dtype() == kFp32) {
    if (a_tensor.shape() == b_tensor.shape()) {
      ew_div_fp32(a_tensor.ptr<float>(), b_tensor.ptr<float>(), c_tensor.ptr<float>(),
                  static_cast<int>(a_tensor.numel()));
      return;
    }

    // broadcast to ele wise.
    // Such as:
    // Tensor a;
    // auto b = a / 1;
    if (b_tensor.shape().size() == 1 && b_tensor.shape()[0] == 1) {
      ew_div_constant_fp32(a_tensor.ptr<float>(), *(b_tensor.ptr<float>()), c_tensor.ptr<float>(),
                           static_cast<int>(a_tensor.numel()));
      return;
    }
  }

  NYI("ArmDivOp::forward op not support current inputs");
}

void ArmNegOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& in = inputs[0];
  auto* out = outputs[0].ptr<void>();
  const int n = static_cast<int>(in.numel());

  switch (in.dtype()) {
    case kFp16: ew_neg_fp16(in.ptr<float16_t>(), reinterpret_cast<float16_t*>(out), n); return;
    case kFp32: ew_neg_fp32(in.ptr<float>(), reinterpret_cast<float*>(out), n); return;
    default: break;
  }

  NYI("ArmNegOp::forward op not support dtype ", in.dtype());
}

}  // namespace mllm::arm
