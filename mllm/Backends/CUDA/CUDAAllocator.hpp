/**
 * @file CUDAAllocator.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Engine/Allocator.hpp"

namespace mllm::cuda {

class CUDAAllocator final : public Allocator {
  bool alloc(const std::shared_ptr<TensorImpl>& tensor) override;
  void free(const std::shared_ptr<TensorImpl>& tensor) override;
  void free(TensorImpl* tensor) override;

  bool generalAlloc(void** ptr, size_t cap, size_t align) override;
  void generalFree(void* ptr) override;

  size_t allocSize(const std::shared_ptr<TensorImpl>& tensor) override;
  size_t allocSize(TensorImpl* tensor) override;
  [[nodiscard]] size_t alignSize() const override;
};

}  // namespace mllm::cuda
