/**
 * @file Allocator.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>
#include <memory>
#include "mllm/Core/Storage.hpp"

namespace mllm {

class Allocator {
 public:
  virtual bool ctrlByMllmMemManager() = 0;
  virtual bool alloc(const std::shared_ptr<Storage>& storage) = 0;
  virtual void free(const std::shared_ptr<Storage>& storage) = 0;
  virtual void free(Storage* storage) = 0;

  virtual bool generalAlloc(void** ptr, size_t cap, size_t align) = 0;
  virtual void generalFree(void* ptr) = 0;

  virtual size_t allocSize(Storage* storage) = 0;
  virtual size_t allocSize(const std::shared_ptr<Storage>& storage) = 0;

  [[nodiscard]] virtual size_t alignSize() const = 0;
};

}  // namespace mllm
