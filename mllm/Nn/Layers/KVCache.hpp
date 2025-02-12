/**
 * @file KVCache.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/KVCacheOp.hpp"

namespace mllm::nn {

// KVCache only accept [B, H, S, D] layout.
// Both inputs and outputs should in [B, H, S, D] layout.
class KVCache : public Layer {
 public:
  KVCache();

  KVCache(size_t heads_num, size_t dim_per_head, size_t head_repeat_times,
          DataTypes cached_elements_dtype, size_t pre_alloc_seq_len,
          size_t re_alloc_multiplier = 2);

  explicit KVCache(const KVCacheOpCargo& cargo);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;
};
}  // namespace mllm::nn
