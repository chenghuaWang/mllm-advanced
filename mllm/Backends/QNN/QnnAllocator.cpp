/**
 * @file QnnAllocator.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstdint>
#include "mllm/Backends/QNN/QnnAllocator.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"

namespace mllm::qnn {

const int QnnAllocator::RPCMEM_HEAP_ID_SYSTEM = 25;

const int QnnAllocator::RPCMEM_DEFAULT_FLAGS = 1;

QnnAllocator::QnnAllocator(const QnnFuncSymbols& qnn_htp_func_symbols,
                           const QnnBackendDevice& qnn_bk_device)
    : qnn_func_symbols_(qnn_htp_func_symbols), qnn_bk_device_(qnn_bk_device) {
  auto& loader = QnnDynSymbolLoader::instance();
  loader.loadQnnDynLib("libcdsprpc.so",
                       QnnDynSymbolLoader::kRTLD_NOW | QnnDynSymbolLoader::kRTLD_LOCAL);

  rpcmem_alloc_func_ = loader("libcdsprpc.so").func<RpcMemAllocFuncType>("rpcmem_alloc");
  rpcmem_free_func_ = loader("libcdsprpc.so").func<RpcMemFreeFuncType>("rpcmem_free");
  rpcmem_to_fd_func_ = loader("libcdsprpc.so").func<RpcMemToFdFuncType>("rpcmem_to_fd");

  MLLM_RT_ASSERT(rpcmem_alloc_func_ != nullptr && rpcmem_free_func_ != nullptr
                 && rpcmem_to_fd_func_ != nullptr);
}

QnnAllocator::~QnnAllocator() {
  for (auto it = qnn_memhandle_map_.begin(); it != qnn_memhandle_map_.end();) {
    MLLM_RT_ASSERT_EQ(QNN_SUCCESS,
                      qnn_func_symbols_.qnn_interface_.memDeRegister(&(it->second.second), 1));
    rpcmem_free_func_(it->first);
    it = qnn_memhandle_map_.erase(it);
  }
}

bool QnnAllocator::alloc(const std::shared_ptr<Storage>& storage) {
  uint8_t* ptr =
      (uint8_t*)rpcmem_alloc_func_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocSize(storage));

  MLLM_RT_ASSERT(ptr != nullptr);

  storage->ptr_ = ptr;
  qnn_mem_set_.insert(ptr);

  return true;
}

void QnnAllocator::free(const std::shared_ptr<Storage>& storage) {
  if (qnn_memhandle_map_.count(storage->ptr_)) {
    MLLM_RT_ASSERT_EQ(QNN_SUCCESS,
                      qnn_func_symbols_.qnn_interface_.memDeRegister(
                          &(qnn_memhandle_map_.find(storage->ptr_)->second.second), 1));
  }

  rpcmem_free_func_(storage->ptr_);
}

void QnnAllocator::free(Storage* storage) {
  if (qnn_memhandle_map_.count(storage->ptr_)) {
    MLLM_RT_ASSERT_EQ(QNN_SUCCESS,
                      qnn_func_symbols_.qnn_interface_.memDeRegister(
                          &(qnn_memhandle_map_.find(storage->ptr_)->second.second), 1));
  }

  rpcmem_free_func_(storage->ptr_);
}

bool QnnAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  MLLM_ERROR_EXIT(kError, "QNN allocator does not support generalAlloc. The generalAlloc is for "
                          "backends that use Mllm's Buddy Memory Manager to use.");
}

void QnnAllocator::generalFree(void* ptr) {
  MLLM_ERROR_EXIT(kError, "QNN allocator does not support generalFree. The generalFree is for "
                          "backends that use Mllm's Buddy Memory Manager to use.");
}

size_t QnnAllocator::allocSize(Storage* storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t QnnAllocator::allocSize(const std::shared_ptr<Storage>& storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t QnnAllocator::alignSize() const {
#if __ARM_ARCH < 8
  return 16;
#else
  return 16;
#endif
}

void QnnAllocator::registerQnnTensorToSharedBuffer(void* ptr, Qnn_Tensor_t& qnn_tensor) {
  // Make sure there has a memory that we can register to.
  MLLM_RT_ASSERT_EQ(qnn_mem_set_.count(ptr), 1);

  // Make sure this memory space is not registered yet.
  MLLM_RT_ASSERT_EQ(qnn_memhandle_map_.count(ptr), 0);

  // Get the file id of this memory space.
  int mem_fd = rpcmem_to_fd_func_(ptr);
  MLLM_RT_ASSERT(mem_fd != -1);

  // Make qnn memory descriptor. Set ION.
  Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
  mem_descriptor.memShape = {
      .numDim = qnn_tensor.v2.rank,
      .dimSize = qnn_tensor.v2.dimensions,
      .shapeConfig = nullptr,
  };
  mem_descriptor.dataType = qnn_tensor.v2.dataType;
  mem_descriptor.memType = QNN_MEM_TYPE_ION;
  mem_descriptor.ionInfo.fd = mem_fd;
  qnn_tensor.v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;

  // Register to QNN memory
  MLLM_RT_ASSERT_EQ(QNN_SUCCESS, qnn_func_symbols_.qnn_interface_.memRegister(
                                     qnn_bk_device_.qnn_ctx_handle_, &mem_descriptor, 1u,
                                     &(qnn_tensor.v2.memHandle)));

  qnn_memhandle_map_.insert({ptr, {mem_fd, qnn_tensor.v2.memHandle}});
}

void QnnAllocator::deRegisterQnnTensorFromSharedBuffer(void* ptr) {
  MLLM_RT_ASSERT_EQ(qnn_memhandle_map_.count(ptr), 1);
  MLLM_RT_ASSERT_EQ(QNN_SUCCESS, qnn_func_symbols_.qnn_interface_.memDeRegister(
                                     &(qnn_memhandle_map_[ptr].second), 1));
  qnn_memhandle_map_.erase(ptr);
}

void QnnAllocator::deRegisterAllQnnTensorFromSharedBuffer() {
  for (auto& kv : qnn_memhandle_map_) {
    MLLM_RT_ASSERT_EQ(QNN_SUCCESS,
                      qnn_func_symbols_.qnn_interface_.memDeRegister(&kv.second.second, 1));
  }
  qnn_memhandle_map_.clear();
}

}  // namespace mllm::qnn
