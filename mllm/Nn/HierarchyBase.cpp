/**
 * @file HierarchyBase.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/HierarchyBase.hpp"

namespace mllm {

HierarchyBase::HierarchyBase(HierarchyTypes type) : type_(type) {}

void HierarchyBase::setName(const std::string& name) { name_ = name; }

void HierarchyBase::setAbsoluteName(const std::string& absolute_name) {
  absolute_name_ = absolute_name;
}

void HierarchyBase::setDepth(int32_t depth) { depth_ = depth; }

void HierarchyBase::depthIncrease() { depth_++; }

void HierarchyBase::depthDecrease() { depth_--; }

std::string HierarchyBase::name() const { return name_; }

std::string HierarchyBase::absoluteName() const { return absolute_name_; }

int32_t HierarchyBase::depth() const { return depth_; }

HierarchyTypes HierarchyBase::type() const { return type_; }

DeviceTypes HierarchyBase::device() const { return device_type_; }

void HierarchyBase::setCompiledAsObj(bool flag) { compiled_as_obj_ = flag; }

bool HierarchyBase::isCompiledAsObj() const { return compiled_as_obj_; }

}  // namespace mllm
