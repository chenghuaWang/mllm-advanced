/**
 * @file CopyOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/CopyOp.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmCopyOp::ArmCopyOp(const CopyOpCargo& cargo) : CopyOp(cargo) {}

void ArmCopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto dst = inputs[0];
  auto src = inputs[1];

  const auto dst_ptr = dst.ptr<char>();
  auto dst_bytes = dst.bytes();

  const auto src_ptr = src.ptr<char>();
  auto src_bytes = src.bytes();

  MLLM_RT_ASSERT_EQ(src_bytes, dst_bytes);

  // Copy all data
  std::memcpy(dst_ptr, src_ptr, dst_bytes);
}

}  // namespace mllm::arm
