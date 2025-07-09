/**
 * @file FlashAttention2Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#include "mllm/Backends/Arm/Ops/FlashAttention2Op.hpp"

namespace mllm::arm {

ArmFlashAttn2Op::ArmFlashAttn2Op(const FlashAttn2OpCargo& cargo) : FlashAttn2Op(cargo) {
  if (cargo.threads == 4 && !cargo.hp_exp && cargo.causal_mask) {
    mobi_attn_kernel_ptr_ = new FA2_4x4_kernel_4_threads_lp_mask();
  } else if (cargo.threads == 4 && !cargo.hp_exp && !cargo.causal_mask) {
    mobi_attn_kernel_ptr_ = new FA2_4x4_kernel_4_threads_lp_nomask();
  } else {
    NYI("Not implemented for threads:{}, hp_exp:{}, causal_mask:{}", cargo.threads, cargo.hp_exp,
        cargo.causal_mask);
  }
}

void ArmFlashAttn2Op::load(const std::shared_ptr<ParameterLoader>& ploader) {
  auto& ctx = MllmEngineCtx::instance();

  if (cargo_.threads == 4 && !cargo_.hp_exp && cargo_.causal_mask) {
    using dtype_t = FA2_4x4_kernel_4_threads_lp_mask::dtype_t;
    using acc_dtype_t = FA2_4x4_kernel_4_threads_lp_mask::acc_dtype_t;

    if (cargo_.hp_exp) {
      MLLM_ERROR_EXIT(
          kError,
          "ArmFlashAttn2Op is instanced for low exp precision for now, use can modify the "
          "FA2_4x4_kernel_4_threads_lp_mask's type to FA2_4x4_kernel_4_threads_hp_mask with `using "
          "FA2_4x4_kernel_4_threads_lp_mask = "
          "mobi_attn::NEON_FA_2_GQA_QKV_FP16_BSHD_O_FP16_BSHD_ACC_FP32_IMPL<4, 4, 4, true, true>;` "
          "in ArmFlashAttn2Op's hpp file.");
    }

    if (ctx.mem()->hasGlobalTensor("__global_acc_s_cast")) {
      ((FA2_4x4_kernel_4_threads_lp_mask*)mobi_attn_kernel_ptr_)
          ->init_workspace(ctx.mem()->getGlobalTensor("__global_acc_s_cast").ptr<dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_o").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_s").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_logsum").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax_prev").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_scale").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_sum").ptr<acc_dtype_t>());
    } else {
      ctx.mem()->regGlobalTensor(Tensor::empty({cargo_.threads, 4, 4}, kFp16, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_acc_s_cast")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty({cargo_.threads, 4, cargo_.D}, kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_acc_o")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty({cargo_.threads, 4, 4}, kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_acc_s")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_logsum")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_scoremax")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_scoremax_prev")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_score_scale")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_score_sum")
                                     .alloc());
      ((FA2_4x4_kernel_4_threads_lp_mask*)mobi_attn_kernel_ptr_)
          ->init_workspace(ctx.mem()->getGlobalTensor("__global_acc_s_cast").ptr<dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_o").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_s").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_logsum").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax_prev").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_scale").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_sum").ptr<acc_dtype_t>());
    }
  } else if (cargo_.threads == 4 && !cargo_.hp_exp && !cargo_.causal_mask) {
    using dtype_t = FA2_4x4_kernel_4_threads_lp_nomask::dtype_t;
    using acc_dtype_t = FA2_4x4_kernel_4_threads_lp_nomask::acc_dtype_t;

    if (ctx.mem()->hasGlobalTensor("__global_acc_s_cast")) {
      ((FA2_4x4_kernel_4_threads_lp_nomask*)mobi_attn_kernel_ptr_)
          ->init_workspace(ctx.mem()->getGlobalTensor("__global_acc_s_cast").ptr<dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_o").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_s").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_logsum").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax_prev").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_scale").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_sum").ptr<acc_dtype_t>());
    } else {
      ctx.mem()->regGlobalTensor(Tensor::empty({cargo_.threads, 4, 4}, kFp16, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_acc_s_cast")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty({cargo_.threads, 4, cargo_.D}, kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_acc_o")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty({cargo_.threads, 4, 4}, kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_acc_s")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_logsum")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_scoremax")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_scoremax_prev")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_score_scale")
                                     .alloc());

      ctx.mem()->regGlobalTensor(Tensor::empty(
                                     {
                                         cargo_.threads,
                                         4,
                                     },
                                     kFp32, kCPU)
                                     .setMemType(kGlobal)
                                     .setName("__global_score_sum")
                                     .alloc());
      ((FA2_4x4_kernel_4_threads_lp_nomask*)mobi_attn_kernel_ptr_)
          ->init_workspace(ctx.mem()->getGlobalTensor("__global_acc_s_cast").ptr<dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_o").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_s").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_logsum").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax_prev").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_scale").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_sum").ptr<acc_dtype_t>());
    }
  }
}

void ArmFlashAttn2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs[0].dtype(), kFp16);
  MLLM_RT_ASSERT_EQ(inputs[1].dtype(), kFp16);
  MLLM_RT_ASSERT_EQ(inputs[2].dtype(), kFp16);

  if (cargo_.threads == 4 && !cargo_.hp_exp && cargo_.causal_mask) {
    using dtype_t = FA2_4x4_kernel_4_threads_lp_mask::dtype_t;
    using acc_dtype_t = FA2_4x4_kernel_4_threads_lp_mask::acc_dtype_t;

    (*((FA2_4x4_kernel_4_threads_lp_mask*)mobi_attn_kernel_ptr_))(
        inputs[0].ptr<dtype_t>(), inputs[1].ptr<dtype_t>(), inputs[2].ptr<dtype_t>(),
        outputs[0].ptr<dtype_t>(), cargo_.B, cargo_.q_head, cargo_.kv_head, inputs[0].shape()[1],
        inputs[1].shape()[1], cargo_.D);
  } else if (cargo_.threads == 4 && !cargo_.hp_exp && !cargo_.causal_mask) {
    using dtype_t = FA2_4x4_kernel_4_threads_lp_nomask::dtype_t;
    using acc_dtype_t = FA2_4x4_kernel_4_threads_lp_nomask::acc_dtype_t;

    (*((FA2_4x4_kernel_4_threads_lp_nomask*)mobi_attn_kernel_ptr_))(
        inputs[0].ptr<dtype_t>(), inputs[1].ptr<dtype_t>(), inputs[2].ptr<dtype_t>(),
        outputs[0].ptr<dtype_t>(), cargo_.B, cargo_.q_head, cargo_.kv_head, inputs[0].shape()[1],
        inputs[1].shape()[1], cargo_.D);
  }
}

}  // namespace mllm::arm
