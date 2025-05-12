/**
 * @file FlashAttn2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/FlashAttention2Op.hpp"

namespace mllm::nn {

class FlashAttn2 : public Layer {
 public:
  FlashAttn2();

  explicit FlashAttn2(const FlashAttn2OpCargo& cargo);

  FlashAttn2(int32_t B, int32_t q_head, int32_t kv_head, int32_t D, int32_t threads = 4,
             bool hp_exp = false, bool causal_mask = true);
};

}  // namespace mllm::nn