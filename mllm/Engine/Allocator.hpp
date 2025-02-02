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
#include "mllm/Core/TensorImpl.hpp"

namespace mllm {

class Allocator {
 public:
  virtual bool alloc(const std::shared_ptr<TensorImpl>& tensor) = 0;
  virtual void free(const std::shared_ptr<TensorImpl>& tensor) = 0;
  virtual void free(TensorImpl* tensor) = 0;

  virtual bool generalAlloc(void** ptr, size_t cap, size_t align) = 0;
  virtual void generalFree(void* ptr) = 0;

  virtual size_t allocSize(TensorImpl* tensor) = 0;
  virtual size_t allocSize(const std::shared_ptr<TensorImpl>& tensor) = 0;
  [[nodiscard]] virtual size_t alignSize() const = 0;
};

}  // namespace mllm
