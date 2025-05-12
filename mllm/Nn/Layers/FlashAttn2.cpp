/**
 * @file FlashAttn2.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/FlashAttn2.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Nn/Layer.hpp"

namespace mllm::nn {

FlashAttn2::FlashAttn2() : Layer(OpType::kFlashAttention_2, FlashAttn2OpCargo{}) {}

FlashAttn2::FlashAttn2(const FlashAttn2OpCargo& cargo) : Layer(OpType::kFlashAttention_2, cargo) {}

FlashAttn2::FlashAttn2(int32_t B, int32_t q_head, int32_t kv_head, int32_t D, int32_t threads,
                       bool hp_exp, bool causal_mask)
    : Layer(OpType::kFlashAttention_2, FlashAttn2OpCargo{
                                           .B = B,
                                           .q_head = q_head,
                                           .kv_head = kv_head,
                                           .D = D,
                                           .threads = threads,
                                           .hp_exp = hp_exp,
                                           .causal_mask = causal_mask,
                                       }) {}

}  // namespace mllm::nn