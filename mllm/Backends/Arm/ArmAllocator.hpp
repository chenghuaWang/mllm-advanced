/**
 * @file ArmAllocator.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Engine/Allocator.hpp"

namespace mllm::arm {

class ArmAllocator final : public Allocator {
 public:
  inline bool ctrlByMllmMemManager() override { return true; }

  bool alloc(const std::shared_ptr<Storage>& storage) override;
  void free(const std::shared_ptr<Storage>& storage) override;
  void free(Storage* storage) override;

  bool generalAlloc(void** ptr, size_t cap, size_t align) override;
  void generalFree(void* ptr) override;

  size_t allocSize(Storage* storage) override;
  size_t allocSize(const std::shared_ptr<Storage>& storage) override;

  [[nodiscard]] size_t alignSize() const override;
};

}  // namespace mllm::arm