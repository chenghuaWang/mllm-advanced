/**
 * @file RepeatOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <arm_neon.h>
#include <algorithm>
#include "mllm/Backends/Arm/Ops/RepeatOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmRepeatOp::ArmRepeatOp(const RepeatOpCargo& cargo) : RepeatOp(cargo) {}

void ArmRepeatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  auto multiplier = cargo_.multiplier;
  auto dim = cargo_.dim;

  if (dim < 0 || dim >= static_cast<int>(X.shape().size())) {
    throw std::invalid_argument("ArmRepeatOp::forward - invalid repeat dimension");
  }

  size_t outer_num = 1;
  for (int i = 0; i < dim; ++i) { outer_num *= X.shape()[i]; }

  size_t dim_size = X.shape()[dim];
  size_t inner_num = 1;
  for (int i = dim + 1; i < X.shape().size(); ++i) { inner_num *= X.shape()[i]; }

  size_t copy_size = inner_num * multiplier;
  size_t x_step = dim_size * inner_num;
  size_t y_step = dim_size * multiplier * inner_num;

  switch (X.dtype()) {
    case kFp32: {
      const float* x_data = X.ptr<float>();
      float* y_data = Y.ptr<float>();

      for (size_t outer = 0; outer < outer_num; ++outer) {
        const float* x_outer_ptr = x_data + outer * x_step;
        float* y_outer_ptr = y_data + outer * y_step;

        for (size_t d = 0; d < dim_size; ++d) {
          const float* src = x_outer_ptr + d * inner_num;
          float* dest = y_outer_ptr + d * multiplier * inner_num;

          for (size_t m = 0; m < multiplier; ++m) {
            std::copy(src, src + inner_num, dest + m * inner_num);
          }
        }
      }
      break;
    }
    case kFp16: {
      const float16_t* x_data = X.ptr<float16_t>();
      float16_t* y_data = Y.ptr<float16_t>();

      for (size_t outer = 0; outer < outer_num; ++outer) {
        const float16_t* x_outer_ptr = x_data + outer * x_step;
        float16_t* y_outer_ptr = y_data + outer * y_step;

        for (size_t d = 0; d < dim_size; ++d) {
          const float16_t* src = x_outer_ptr + d * inner_num;
          float16_t* dest = y_outer_ptr + d * multiplier * inner_num;

          for (size_t m = 0; m < multiplier; ++m) {
            std::copy(src, src + inner_num, dest + m * inner_num);
          }
        }
      }
      break;
    }
    default: NYI("ArmRepeatOp::forward not support dtype {}", dataTypes2Str(X.dtype())); break;
  }
}

}  // namespace mllm::arm