/**
 * @file Storage.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>

#include "mllm/Core/DeviceTypes.hpp"

namespace mllm {

class Storage : public std::enable_shared_from_this<Storage> {
 public:
  virtual ~Storage() = default;

  enum StorageTypes : uint8_t {
    kStart = 0,
    kBase,
    kTensor,
    kEnd,
  };

  void* ptr_ = nullptr;
  size_t size_ = 0;
  DeviceTypes device_ = kCPU;
  StorageTypes type_ = kBase;
  uint32_t custom_32bit_uuid_ = -1;
};

}  // namespace mllm
