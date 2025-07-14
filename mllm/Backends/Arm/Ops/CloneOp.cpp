/**
 * @file CloneOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/CloneOp.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmCloneOp::ArmCloneOp(const CloneOpCargo& cargo) : CloneOp(cargo) {}

void ArmCloneOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto dst = outputs[0];
  auto src = inputs[0];

  const auto dst_ptr = dst.ptr<char>();
  auto dst_bytes = dst.bytes();

  const auto src_ptr = src.ptr<char>();
  auto src_bytes = src.bytes();

  MLLM_RT_ASSERT_EQ(src_bytes, dst_bytes);

  // Copy all data
  std::memcpy(dst_ptr, src_ptr, dst_bytes);
}

}  // namespace mllm::arm
