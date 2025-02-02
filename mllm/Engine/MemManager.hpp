/**
 * @file MemManager.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <map>
#include <list>
#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include <vector>
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Engine/Allocator.hpp"
#include "mllm/Engine/SymbolTable.hpp"

namespace mllm {

struct ObjMemSegment;

struct ObjMemBlock {
  char* ptr = nullptr;
  size_t offset = 0;
  size_t size = 0;
  ObjMemSegment* segment = nullptr;
  size_t buddy_order = 0;  // solver specific data
  bool allocated = false;
};

struct ObjMemSegment {
  char* ptr = nullptr;
  size_t cap = 0;
  size_t used = 0;
  size_t min_order = 0;
  size_t max_order = 0;
};

struct MemManagerCargo {
  // buddy related
  size_t buddy_first_segment_cap = 128 * 1024 * 1024;  // 128MB
  size_t buddy_min_order = 14;
  size_t buddy_max_order = 27;

  // threshold
  size_t really_large_tensor_threshold = 128 * 1024 * 1024;  // 128MB

  // cache related
  std::vector<size_t> cache_size_list{};

  // clean up periodically
  size_t clean_up_period = 5000;  // ms
};

struct BuddyCtx {
  std::map<char*, ObjMemSegment*> segments;  // sorted by ptr.
  std::map<char*, std::vector<std::list<ObjMemBlock*>>> segment_blocks;
};

class MemManager {
 public:
  ~MemManager();
  explicit MemManager(MemManagerCargo cargo);

  void alloc(const std::shared_ptr<TensorImpl>&);

  void free(TensorImpl* t);

  void regAllocator(DeviceTypes device, const std::shared_ptr<Allocator>& allocator);

  void updateCacheSizeList(DeviceTypes device, const std::vector<size_t>& cache_size_list);

  void report() const;

  void initBuddyCtx(DeviceTypes device);

  void initOC(DeviceTypes device);

 private:
  ObjMemBlock* _alloc_buddy(DeviceTypes device, size_t omb_size);
  void _free_buddy(DeviceTypes device, ObjMemBlock* omb);
  // this function is not thread safe, should be called in a thread safe context.
  bool _expand_buddy_segment(DeviceTypes device);

  ObjMemBlock* _alloc_oc(DeviceTypes device, size_t omb_size);
  void _free_oc(DeviceTypes device, ObjMemBlock* omb);
  bool _is_in_oc(size_t omb_size);

  void _alloc_really_large(const std::shared_ptr<TensorImpl>&);
  void _free_really_large(TensorImpl* t);

  // this function is not thread safe, should be called in a thread safe context.
  ObjMemSegment* _locate_segment(DeviceTypes device, char* ptr);

  void cleanUp();

  // buddy algorithm related
  SymbolTable<DeviceTypes, BuddyCtx*> buddy_ctx_st_;

  // object cache related
  int32_t oc_hit_times_ = 0;
  int32_t oc_cache_times_ = 0;
  SymbolTable<DeviceTypes, std::unordered_map<size_t, std::list<ObjMemBlock*>>>
      free_object_cache_st_;

  // cargo
  MemManagerCargo cargo_;

  // all allocators
  SymbolTable<DeviceTypes, std::shared_ptr<Allocator>> allocators_st_;

  // symbol table.
  SymbolTable<uint32_t, ObjMemBlock*> st_;
  SymbolTable<std::string, ObjMemBlock*> named_tensor_st_;

  // make manager thread safe.
  std::mutex mutex_;
  std::atomic<bool> cleanup_thread_running_{false};
  std::thread cleanup_thread_;
};

}  // namespace mllm
