/**
 * @file BaseOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm {

BaseOp::BaseOp(OpType op_type) : op_type_(op_type) {}

std::string BaseOp::name() const { return name_; }

void BaseOp::setName(const std::string& name) { name_ = name; }

DeviceTypes BaseOp::device() const { return device_type_; }

void BaseOp::setDeviceType(DeviceTypes device_type) { device_type_ = device_type; }

void BaseOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& TCB = MllmEngineCtx::instance().thisThread()->getTCB();
  auto& registered_outs = TCB.planning_ctx_.registered_outs_;

  std::vector<uint8_t> alloced_flags(outputs.size(), 0);

  // Handle registered buffer here. The tensor in registered buffer is already alloced.
  for (const auto& [tensor, pos] : registered_outs) {
    // Replace current tensor if shape is match, else panic.
    MLLM_RT_ASSERT(pos < outputs.size());
    MLLM_RT_ASSERT_EQ(outputs[pos].shape(), tensor.shape());
    MLLM_RT_ASSERT_EQ(outputs[pos].dtype(), tensor.dtype());
    MLLM_RT_ASSERT_EQ(outputs[pos].device(), tensor.device());
    outputs[pos] = tensor;
    alloced_flags[pos] = 1;  // Mark as alloced
  }

  // Alloc what we need to alloc
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (alloced_flags[i] == 0) { outputs[i].alloc(); }
  }
}

}  // namespace mllm
