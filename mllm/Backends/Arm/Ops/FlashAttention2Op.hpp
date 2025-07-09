/**
 * @file FlashAttention2Op.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/FlashAttention2Op.hpp"
#include "mobi_attn/flash_attn_2/driver.hpp"
#include "mobi_attn/flash_attn_2/arm_qkv_fp16_o_fp16_fa2.hpp"

namespace mllm::arm {

class ArmFlashAttn2Op final : public FlashAttn2Op {
  // Only 4x4 kernel is highly optimized in MobiAttn
  using FA2_4x4_kernel_4_threads_lp_mask = mobi_attn::FlashAttn2<
      mobi_attn::NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL<4, 4, 4, false, true>>;

  using FA2_4x4_kernel_4_threads_lp_nomask = mobi_attn::FlashAttn2<
      mobi_attn::NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL<4, 4, 4, false, false>>;

 public:
  explicit ArmFlashAttn2Op(const FlashAttn2OpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  void* mobi_attn_kernel_ptr_ = nullptr;
};

class ArmFlashAttn2OpFactory final
    : public TypedOpFactory<OpType::kFlashAttention_2, FlashAttn2OpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const FlashAttn2OpCargo& cargo) override {
    return std::make_shared<ArmFlashAttn2Op>(cargo);
  }
};

}  // namespace mllm::arm