/**
 * @file KVCache.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/KVCache.hpp"

namespace mllm::nn {

KVCache::KVCache() : Layer(OpType::kKVCache, KVCacheOpCargo{}) {}

KVCache::KVCache(int32_t heads_num, int32_t dim_per_head, int32_t head_repeat_times,
                 DataTypes cached_elements_dtype, int32_t pre_alloc_seq_len,
                 int32_t re_alloc_multiplier)
    : Layer(OpType::kKVCache, KVCacheOpCargo{
                                  .heads_num = heads_num,
                                  .dim_per_head = dim_per_head,
                                  .head_repeat_times = head_repeat_times,
                                  .cached_elements_dtype = cached_elements_dtype,
                                  .pre_alloc_seq_len = pre_alloc_seq_len,
                                  .re_alloc_multiplier = re_alloc_multiplier,
                                  .layout_type = KVCacheOpCargo::KVCacheLayoutType::kBHSD_REPEAT,
                              }) {}

KVCache::KVCache(int32_t heads_num, int32_t dim_per_head, int32_t head_repeat_times,
                 DataTypes cached_elements_dtype, int32_t pre_alloc_seq_len,
                 KVCacheOpCargo::KVCacheLayoutType layout_type, int32_t re_alloc_multiplier)
    : Layer(OpType::kKVCache, KVCacheOpCargo{
                                  .heads_num = heads_num,
                                  .dim_per_head = dim_per_head,
                                  .head_repeat_times = head_repeat_times,
                                  .cached_elements_dtype = cached_elements_dtype,
                                  .pre_alloc_seq_len = pre_alloc_seq_len,
                                  .re_alloc_multiplier = re_alloc_multiplier,
                                  .layout_type = layout_type,
                              }) {
  if (layout_type == KVCacheOpCargo::KVCacheLayoutType::kBSHD_NO_REPEAT && head_repeat_times != 1) {
    MLLM_ERROR_EXIT(kError, "layout_type == KVCacheOpCargo::KVCacheLayoutType::kBSHD_NO_REPEAT &&  "
                            "head_repeat_times != 1  CHECK FAILED.");
  }
}

}  // namespace mllm::nn
