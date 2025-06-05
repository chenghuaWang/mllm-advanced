/**
 * @file QnnAllocator.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <set>
#include <functional>
#include <unordered_map>
#include "mllm/Utils/Common.hpp"
#include "mllm/Engine/Allocator.hpp"
#include "mllm/Backends/QNN/Runtime/QnnCommon.hpp"

namespace mllm::qnn {

class QnnAllocator final : public Allocator {
  using RpcMemAllocFuncType = void*(int, uint32_t, int);
  using RpcMemFreeFuncType = void(void*);
  using RpcMemToFdFuncType = int(void*);

 public:
  QnnAllocator(const QnnFuncSymbols& qnn_htp_func_symbols, const QnnBackendDevice& qnn_bk_device);

  ~QnnAllocator();

  // Mllm memory manager should not take control QNN's memory. QNN use ION on android to manage all
  // memory. The main purpose of Android's ION subsystem is to achieve zero-copy shared memory
  // between devices by allocating and sharing memory between hardware devices and user space.
  inline bool ctrlByMllmMemManager() override { return false; }

  bool alloc(const std::shared_ptr<Storage>& storage) override;

  void free(const std::shared_ptr<Storage>& storage) override;

  void free(Storage* storage) override;

  bool generalAlloc(void** ptr, size_t cap, size_t align) override;

  void generalFree(void* ptr) override;

  size_t allocSize(Storage* storage) override;

  size_t allocSize(const std::shared_ptr<Storage>& storage) override;

  [[nodiscard]] size_t alignSize() const override;

  // Sharing access in between processing domains in QNN HTP backend. Using shared buffers can
  // eliminate data copy in between client code on the host CPU and HTP accelerator.
  void registerQnnTensorToSharedBuffer(void* ptr, Qnn_Tensor_t& qnn_tensor);

  void deRegisterQnnTensorFromSharedBuffer(void* ptr);

  void deRegisterAllQnnTensorFromSharedBuffer();

 private:
  static const int RPCMEM_HEAP_ID_SYSTEM;
  static const int RPCMEM_DEFAULT_FLAGS;

  std::set<void*> qnn_mem_set_;
  const QnnFuncSymbols& qnn_func_symbols_;
  const QnnBackendDevice& qnn_bk_device_;

  // To allocate and free ION memory.
  std::function<RpcMemAllocFuncType> rpcmem_alloc_func_ = nullptr;
  std::function<RpcMemFreeFuncType> rpcmem_free_func_ = nullptr;

  // To obtain a file descriptor that refers to allocated memory, which can be registered with
  // a backend via QnnMem_register()
  std::function<RpcMemToFdFuncType> rpcmem_to_fd_func_ = nullptr;
  std::unordered_map<void*, std::pair<int, Qnn_MemHandle_t>> qnn_memhandle_map_;
};

}  // namespace mllm::qnn
