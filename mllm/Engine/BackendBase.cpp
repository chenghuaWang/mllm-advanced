/**
 * @file BackendBase.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/BackendBase.hpp"

namespace mllm {

BackendBase::BackendBase(DeviceTypes device) : device_(device) {}

std::shared_ptr<Allocator> BackendBase::getAllocator() const { return allocator_; }

std::shared_ptr<BaseOp> BackendBase::createOp(OpType op_type, const BaseOpCargoBase& base_cargo) {
  auto op = op_factory_table_[op_type]->create(base_cargo);
  op->setDeviceType(device_);
  return op;
}

DeviceTypes BackendBase::deviceType() const { return device_; }

}  // namespace mllm
