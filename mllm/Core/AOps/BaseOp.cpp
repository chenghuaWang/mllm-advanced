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

namespace mllm {

BaseOp::BaseOp(OpType op_type) : op_type_(op_type) {}

std::string BaseOp::name() const { return name_; }

void BaseOp::setName(const std::string& name) { name_ = name; }

DeviceTypes BaseOp::device() const { return device_type_; }

void BaseOp::setDeviceType(DeviceTypes device_type) { device_type_ = device_type; }

}  // namespace mllm
