/**
 * @file BackendBase.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Engine/Allocator.hpp"
#include "mllm/Engine/SymbolTable.hpp"

namespace mllm {

class BackendBase {
 public:
  explicit BackendBase(DeviceTypes device);

  template<typename... Args>
  void regOpFactory() {
    (..., (_reg_one_op_factory<Args>()));
  }

  [[nodiscard]] std::shared_ptr<Allocator> getAllocator() const;

  std::shared_ptr<BaseOp> createOp(OpType op_type, const BaseOpCargoBase& base_cargo);

  [[nodiscard]] DeviceTypes deviceType() const;

 private:
  template<typename T>
  void _reg_one_op_factory() {
    auto ptr = std::make_shared<T>();
    op_factory_table_.reg(ptr->opType(), ptr);
  }

 protected:
  DeviceTypes device_;
  SymbolTable<OpType, std::shared_ptr<BaseOpFactory>> op_factory_table_;
  std::shared_ptr<Allocator> allocator_ = nullptr;
};

}  // namespace mllm