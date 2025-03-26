/**
 * @file MemManager.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cmath>
#include <algorithm>
#include <memory>
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Engine/MemManager.hpp"

namespace mllm {

namespace {
size_t _log2_ceil(size_t size) {
  if (size == 0) return 0;
  return static_cast<size_t>(ceil(log2(size)));
}

}  // namespace

MemManager::~MemManager() {
  for (auto& kv : buddy_ctx_st_._raw_data()) {
    auto buddy_ctx = kv.second;
    for (auto& seg : buddy_ctx->segments) {
      auto allocator = allocators_st_[kv.first];
      allocator->generalFree(seg.second->ptr);
    }
  }
}

MemManager::MemManager(MemManagerCargo cargo) : cargo_(std::move(cargo)) {}

void MemManager::alloc(const std::shared_ptr<Storage>& s) {
  auto allocator = allocators_st_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  // large storage
  if (try_to_alloc_size >= cargo_.really_large_tensor_threshold) {
    MLLM_WARN("Trying to alloc a really large storage, whose storage size is {}B. The mllm memory "
              "manager will alloc a memory for this storage from OS directly instead of "
              "allocating one from ObjectCachePool/BuddyMemoryPool. If your scenario need to "
              "handle large storage frequently, you can modify the `buddy_first_segment_cap` in "
              "`MemManagerCargo`.",
              try_to_alloc_size);
    _alloc_really_large(s);
    return;
  }

  // object cache
  ObjMemBlock* omb = nullptr;
  if (_is_in_oc(try_to_alloc_size)) { omb = _alloc_oc(s->device_, try_to_alloc_size); }

  // buddy
  if (!omb) { omb = _alloc_buddy(s->device_, try_to_alloc_size); }

  s->ptr_ = omb->ptr;

  st_.reg(s->custom_32bit_uuid_, omb);

  if (s->type_ == Storage::kTensor && !std::static_pointer_cast<TensorStorage>(s)->name_.empty()) {
    named_tensor_st_.reg(std::static_pointer_cast<TensorStorage>(s)->name_, omb);
  }
}

void MemManager::free(Storage* s) {
  auto allocator = allocators_st_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  if (try_to_alloc_size >= cargo_.really_large_tensor_threshold) {
    _free_really_large(s);
    return;
  }

  if (_is_in_oc(try_to_alloc_size)) {
    _free_oc(s->device_, st_[s->custom_32bit_uuid_]);
  } else {
    _free_buddy(s->device_, st_[s->custom_32bit_uuid_]);
  }

  st_.remove(s->custom_32bit_uuid_);
  if (s->type_ == Storage::kTensor && !((TensorStorage*)(s))->name_.empty()) {
    named_tensor_st_.remove(((TensorStorage*)(s))->name_);
  }
}

void MemManager::regAllocator(DeviceTypes device, const std::shared_ptr<Allocator>& allocator) {
  allocators_st_.reg(device, allocator);
}

void MemManager::updateCacheSizeList(DeviceTypes device,
                                     const std::unordered_set<size_t>& cache_size_set) {
  auto& object_cache = free_object_cache_st_[device];

  for (auto size : cache_size_set) {
    if (!_is_in_oc(size)) {
      auto& free_list = object_cache[size];
      for (auto it : free_list) { _free_buddy(device, it); }
      object_cache.erase(object_cache.find(size));
    }
  }

  cargo_.cache_size_set = cache_size_set;
}

void MemManager::report() const {
  MLLM_INFO("Object Memory Hit Times: {}", oc_hit_times_);
  MLLM_INFO("Object Memory Cached Times: {}", oc_cache_times_);
  for (auto& kv : buddy_ctx_st_._raw_data()) {
    MLLM_INFO("Memory Pool of device -> {}", deviceTypes2Str(kv.first));
    auto buddy_ctx = kv.second;
    for (auto& seg : buddy_ctx->segments) {
      MLLM_INFO("address: {:#010x}, cap: {}B, used: {}B", (uintptr_t)seg.first, seg.second->cap,
                seg.second->used);
    }
  }
}

void MemManager::initBuddyCtx(DeviceTypes device) {
  // MemManager will init
  auto default_device = device;
  if (buddy_ctx_st_.has(default_device)) {
    MLLM_ERROR_EXIT(kError,
                    "Double init buddy ctx of device type {} is not allowed in mllm. You can push "
                    "a feature request to mllm-advanced repo.",
                    deviceTypes2Str(default_device));
  }

  // align to 4KB
  void* _p = nullptr;
  allocators_st_[default_device]->generalAlloc(&_p, cargo_.buddy_first_segment_cap, 4096);

  auto new_seg = new ObjMemSegment{
      .ptr = (char*)_p,
      .cap = cargo_.buddy_first_segment_cap,
      .used = 0,
      .min_order = cargo_.buddy_min_order,
      .max_order = cargo_.buddy_max_order,
  };

  MLLM_RT_ASSERT_EQ(_log2_ceil(cargo_.buddy_first_segment_cap), cargo_.buddy_max_order);
  auto buddy_ctx = new BuddyCtx();
  buddy_ctx->segments.insert({new_seg->ptr, new_seg});
  buddy_ctx->segment_blocks.insert(
      {new_seg->ptr,
       std::vector<std::list<ObjMemBlock*>>(cargo_.buddy_max_order - cargo_.buddy_min_order + 1)});
  buddy_ctx_st_.reg(default_device, buddy_ctx);

  auto block = new ObjMemBlock{
      .ptr = new_seg->ptr,
      .offset = 0,
      .size = cargo_.buddy_first_segment_cap,
      .segment = new_seg,
      .buddy_order = cargo_.buddy_max_order,
      .allocated = false,
  };

  buddy_ctx->segment_blocks[new_seg->ptr][cargo_.buddy_max_order - cargo_.buddy_min_order]
      .push_back(block);
}

void MemManager::initOC(DeviceTypes device) { free_object_cache_st_.reg(device, {}); }

void MemManager::regGlobalTensor(Tensor v) { global_tensor_st_.reg(v.name(), v); }

Tensor MemManager::getGlobalTensor(const std::string& name) { return global_tensor_st_[name]; }

bool MemManager::hasGlobalTensor(const std::string& name) { return global_tensor_st_.has(name); }

void MemManager::clearGlobalTensor() { global_tensor_st_._ref_raw_data().clear(); }

ObjMemBlock* MemManager::_alloc_buddy(DeviceTypes device, size_t omb_size) {
  // lock_guard should in this scope.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& cur_segs = buddy_ctx_st_[device]->segments;
    auto& cur_seg_blocks = buddy_ctx_st_[device]->segment_blocks;

    // loop all seg.
    for (auto seg : cur_segs) {
      auto& free_lists = cur_seg_blocks[seg.first];
      auto min_order = seg.second->min_order;
      auto max_order = seg.second->max_order;

      size_t required_size = std::max(omb_size, (size_t)(1ULL << min_order));
      if (required_size > (seg.second->cap - seg.second->used)) continue;

      size_t order = _log2_ceil(required_size);

      if (order > max_order) {
        MLLM_ERROR_EXIT(
            kError,
            "The tensor size {} you want to alloc is too large. Buddy memory pool support max "
            "tensor "
            "size is {}. You should change the `buddy_first_segment_cap` in MemManagerCargo.",
            required_size, cargo_.buddy_first_segment_cap);
      }
      if (order < min_order) order = min_order;

      // search for usable order
      auto current_order = order;
      while (current_order <= max_order) {
        size_t idx = current_order - min_order;
        if (idx >= free_lists.size() || free_lists[idx].empty()) {
          current_order++;
          continue;
        }

        // get the empty block
        auto block = free_lists[idx].front();
        free_lists[idx].pop_front();

        // split this empty block if it has larger order
        while (block->buddy_order > order) {
          size_t new_size = block->size / 2;
          block->size = new_size;
          block->buddy_order--;

          // create block's buddy
          auto buddy = new ObjMemBlock{
              .ptr = block->ptr + new_size,
              .offset = block->offset + new_size,
              .size = new_size,
              .segment = block->segment,
              .buddy_order = block->buddy_order,  // solver specific data
              .allocated = false,
          };

          free_lists[block->buddy_order - min_order].push_back(buddy);
        }

        block->allocated = true;

        seg.second->used += block->size;

        return block;
      }
    }
  }
  // lock_guard freed, so that we can call _alloc_buddy again
  // if not found, alloc a new seg.
  if (_expand_buddy_segment(device)) { return _alloc_buddy(device, omb_size); }

  MLLM_ERROR_EXIT(kError, "Failed to alloc a new segment for buddy memory pool.");

  return nullptr;
}

void MemManager::_free_buddy(DeviceTypes device, ObjMemBlock* omb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto cur_seg = omb->segment;
  auto& cur_seg_blocks = buddy_ctx_st_[device]->segment_blocks[cur_seg->ptr];

  size_t min_order = cur_seg->min_order;
  size_t max_order = cur_seg->max_order;

  omb->allocated = false;
  cur_seg->used -= omb->size;

  // joint buddy blocks
  while (true) {
    size_t current_order = omb->buddy_order;
    size_t buddy_size = (1ULL << current_order);

    auto buddy_addr = (size_t)(omb->ptr - cur_seg->ptr) ^ buddy_size;

    auto& list = cur_seg_blocks[current_order - min_order];
    auto it = std::find_if(list.begin(), list.end(), [&](const ObjMemBlock* b) {
      return (b->ptr - cur_seg->ptr) == buddy_addr && b->buddy_order == current_order
             && !b->allocated;
    });

    if (it == list.end()) {
      list.push_back(omb);
      break;
    }

    if (omb->ptr < (*it)->ptr) {
      omb->size *= 2;
      omb->buddy_order++;
    } else {
      omb->ptr = (*it)->ptr;
      omb->size *= 2;
      omb->buddy_order++;
    }
    list.erase(it);
    delete (*it);
  }
}

bool MemManager::_expand_buddy_segment(DeviceTypes device) {
  auto buddy_ctx = buddy_ctx_st_[device];
  auto& cur_segs = buddy_ctx->segments;
  auto& cur_seg_blocks = buddy_ctx->segment_blocks;

  size_t previous_seg_cap = 0;
  for (auto& seg : cur_segs) {
    previous_seg_cap = std::max(previous_seg_cap, (size_t)(1ULL << seg.second->max_order));
  }

  size_t min_order = cargo_.buddy_min_order;
  size_t new_cap = std::min(previous_seg_cap * 2, (size_t)(1ULL << 29ULL));  // max is 512MB
  size_t max_order = _log2_ceil(new_cap);

  void* _p = nullptr;
  if (!allocators_st_[device]->generalAlloc(&_p, new_cap, 4096)) { return false; }

  auto new_seg = new ObjMemSegment{
      .ptr = (char*)_p,
      .cap = new_cap,
      .used = 0,
      .min_order = min_order,
      .max_order = max_order,
  };

  cur_segs.insert({new_seg->ptr, new_seg});
  cur_seg_blocks.insert(
      {new_seg->ptr, std::vector<std::list<ObjMemBlock*>>(max_order - min_order + 1)});

  auto block = new ObjMemBlock{
      .ptr = new_seg->ptr,
      .offset = 0,
      .size = new_cap,
      .segment = new_seg,
      .buddy_order = max_order,
      .allocated = false,
  };
  cur_seg_blocks[new_seg->ptr][max_order - min_order].push_back(block);

  return true;
}

ObjMemBlock* MemManager::_alloc_oc(DeviceTypes device, size_t omb_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& object_cache = free_object_cache_st_[device];

  if (!object_cache.count(omb_size) || object_cache[omb_size].empty()) return nullptr;

  auto ret = object_cache[omb_size].front();
  object_cache[omb_size].pop_front();

  oc_hit_times_++;

  return ret;
}

void MemManager::_free_oc(DeviceTypes device, ObjMemBlock* omb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& object_cache = free_object_cache_st_[device];
  object_cache[omb->size].push_back(omb);
  oc_cache_times_++;
}

bool MemManager::_is_in_oc(size_t omb_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  return cargo_.cache_size_set.count(omb_size);
}

void MemManager::_alloc_really_large(const std::shared_ptr<Storage>& s) {
  auto allocator = allocators_st_[s->device_];
  allocator->alloc(s);

  // register to memory manager
  auto obj_mem_block = new ObjMemBlock{
      .ptr = (char*)s->ptr_,
      .offset = 0,
      .size = 0,
      .segment = nullptr,
      .buddy_order = 0,
      .allocated = true,
  };
  if (s->type_ == Storage::kTensor && !std::static_pointer_cast<TensorStorage>(s)->name_.empty()) {
    named_tensor_st_.reg(std::static_pointer_cast<TensorStorage>(s)->name_, obj_mem_block);
  }

  st_.reg(s->custom_32bit_uuid_, obj_mem_block);
}

void MemManager::_free_really_large(Storage* s) {
  auto allocator = allocators_st_[s->device_];
  allocator->free(s);

  auto obj_mem_block = st_[s->custom_32bit_uuid_];
  st_.remove(s->custom_32bit_uuid_);
  if (s->type_ == Storage::kTensor && !((TensorStorage*)(s))->name_.empty()) {
    named_tensor_st_.remove(((TensorStorage*)(s))->name_);
  }
  delete obj_mem_block;
}

ObjMemSegment* MemManager::_locate_segment(DeviceTypes device, char* ptr) {
  auto& segment = buddy_ctx_st_[device]->segments;
  auto it = segment.upper_bound(ptr);
  if (it == segment.begin()) return nullptr;
  --it;
  auto seg = it->second;
  if (ptr >= seg->ptr && ptr < seg->ptr + seg->cap) { return seg; }
  return nullptr;
}

void MemManager::cleanUp() {
  // TODO
}

}  // namespace mllm