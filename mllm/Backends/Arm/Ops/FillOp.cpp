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

namespace mllm::arm {

namespace {
void __iteration_copy_tensor(Tensor& dst, Tensor& src, size_t ele_size,
                             std::vector<size_t>& indices, std::vector<size_t>& shape,
                             size_t dim = 0) {
  if (dim == indices.size()) {
    std::memcpy(dst.offsettedRawPtr(indices), src.offsettedRawPtr(indices), ele_size);
    return;
  }

  for (size_t i = 0; i < shape[dim]; ++i) {
    indices[dim] = i;
    __iteration_copy_tensor(dst, src, ele_size, indices, shape, dim + 1);
  }
}
}  // namespace

ArmFillOp::ArmFillOp(const FillOpCargo& cargo) : FillOp(cargo) {}

void ArmFillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& t = inputs[0];
  auto dtype = t.dtype();
  // type
  // 0 -> zeros
  // 1 -> ones
  // 2 -> specific
  // 3 -> random
  // 4 -> arrange
  // 5 -> make tensor contiguous
  switch (cargo_.type) {
    case 0: {
      MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
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
      MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
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
    case 2: MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid()); break;
    case 3: MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid()); break;
    case 4: MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid()); break;
    case 5: {
      auto t = inputs[0];
      auto o = outputs[0];
      auto shape = t.shape();
      auto indicies = std::vector<size_t>(shape.size(), 0);
      __iteration_copy_tensor(o, t, dataTypeSize(t.dtype()), indicies, shape, 0);
      break;
    }
    default:
      MLLM_WARN("ArmFillOp found cargo.type={}, which is not supported yet. The ArmFillOp will do "
                "nothing on input tensor.",
                cargo_.type);
      break;
  }
}

}  // namespace mllm::arm
