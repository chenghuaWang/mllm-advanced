/**
 * @file HierarchyBase.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "mllm/Core/DeviceTypes.hpp"

namespace mllm {

enum class HierarchyTypes : int32_t {
  kModule = 0,
  kLayer,
};

class HierarchyBase : public std::enable_shared_from_this<HierarchyBase> {
 public:
  explicit HierarchyBase(HierarchyTypes type);

  void setName(const std::string& name);

  void setAbsoluteName(const std::string& absolute_name);

  void setDepth(int32_t depth);

  void depthIncrease();

  void depthDecrease();

  [[nodiscard]] std::string name() const;

  [[nodiscard]] std::string absoluteName() const;

  [[nodiscard]] int32_t depth() const;

  [[nodiscard]] HierarchyTypes type() const;

  [[nodiscard]] DeviceTypes device() const;

  void setCompiledAsObj(bool flag);

  [[nodiscard]] bool isCompiledAsObj() const;

 protected:
  bool compiled_as_obj_ = false;
  HierarchyTypes type_;
  int32_t depth_ = 0;
  DeviceTypes device_type_ = kCPU;
  std::string name_;
  std::string absolute_name_;
};

}  // namespace mllm
