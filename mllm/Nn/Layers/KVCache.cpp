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

KVCache::KVCache(size_t heads_num, size_t dim_per_head, size_t head_repeat_times,
                 DataTypes cached_elements_dtype, size_t pre_alloc_seq_len,
                 size_t re_alloc_multiplier)
    : Layer(OpType::kKVCache, KVCacheOpCargo{
                                  .heads_num = heads_num,
                                  .dim_per_head = dim_per_head,
                                  .head_repeat_times = head_repeat_times,
                                  .cached_elements_dtype = cached_elements_dtype,
                                  .pre_alloc_seq_len = pre_alloc_seq_len,
                                  .re_alloc_multiplier = re_alloc_multiplier,
                              }) {}

}  // namespace mllm::nn
