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
#include "mllm/Utils/Dbg.hpp"
#include "mllm/Backends/Arm/Ops/FlashAttention2Op.hpp"

namespace mllm::arm {

ArmFlashAttn2Op::ArmFlashAttn2Op(const FlashAttn2OpCargo& cargo) : FlashAttn2Op(cargo) {
  if (cargo_.q_head == 32 && cargo_.kv_head == 8 && cargo_.threads == 4 && !cargo_.hp_exp) {
    mobi_attn_kernel_ptr_ = new DEF_FA2_NAME(32, 8, 4, false)();
  } else if (cargo_.q_head == 12 && cargo_.kv_head == 2 && cargo_.threads == 4 && !cargo_.hp_exp) {
    mobi_attn_kernel_ptr_ = new DEF_FA2_NAME(12, 2, 4, false)();
  } else {
    MLLM_ERROR_EXIT(kError,
                    "cargo_.q_head({}), cargo_.kv_head({}), cargo_.threads({}), cargo_.hp_exp({}) "
                    "is not predefined in FA2 Op",
                    cargo_.q_head, cargo_.kv_head, cargo_.threads, cargo_.hp_exp);
  }
}

void ArmFlashAttn2Op::load(const std::shared_ptr<ParameterLoader>& ploader) {
  auto& ctx = MllmEngineCtx::instance();

  if (cargo_.q_head == 32 && cargo_.kv_head == 8 && cargo_.threads == 4 && !cargo_.hp_exp) {
    using dtype_t = DEF_FA2_NAME(32, 8, 4, false)::dtype_t;
    using acc_dtype_t = DEF_FA2_NAME(32, 8, 4, false)::acc_dtype_t;
    if (ctx.mem()->hasGlobalTensor("__global_acc_s_cast")) {
      ((DEF_FA2_NAME(32, 8, 4, false)*)mobi_attn_kernel_ptr_)
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
      ((DEF_FA2_NAME(32, 8, 4, false)*)mobi_attn_kernel_ptr_)
          ->init_workspace(ctx.mem()->getGlobalTensor("__global_acc_s_cast").ptr<dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_o").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_s").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_logsum").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax_prev").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_scale").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_sum").ptr<acc_dtype_t>());
    }

  } else if (cargo_.q_head == 12 && cargo_.kv_head == 2 && cargo_.threads == 4 && !cargo_.hp_exp) {
    using dtype_t = DEF_FA2_NAME(12, 2, 4, false)::dtype_t;
    using acc_dtype_t = DEF_FA2_NAME(12, 2, 4, false)::acc_dtype_t;

    if (ctx.mem()->hasGlobalTensor("__global_acc_s_cast")) {
      ((DEF_FA2_NAME(12, 2, 4, false)*)mobi_attn_kernel_ptr_)
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

      ((DEF_FA2_NAME(12, 2, 4, false)*)mobi_attn_kernel_ptr_)
          ->init_workspace(ctx.mem()->getGlobalTensor("__global_acc_s_cast").ptr<dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_o").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_acc_s").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_logsum").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_scoremax_prev").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_scale").ptr<acc_dtype_t>(),
                           ctx.mem()->getGlobalTensor("__global_score_sum").ptr<acc_dtype_t>());
    }

  } else {
    MLLM_ERROR_EXIT(kError,
                    "cargo_.q_head({}), cargo_.kv_head({}), cargo_.threads({}), cargo_.hp_exp({}) "
                    "is not predefined in FA2 Op",
                    cargo_.q_head, cargo_.kv_head, cargo_.threads, cargo_.hp_exp);
  }
}

void ArmFlashAttn2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (cargo_.q_head == 32 && cargo_.kv_head == 8 && cargo_.threads == 4 && !cargo_.hp_exp) {
    using dtype_t = DEF_FA2_NAME(32, 8, 4, false)::dtype_t;
    using acc_dtype_t = DEF_FA2_NAME(32, 8, 4, false)::acc_dtype_t;

    (*((DEF_FA2_NAME(32, 8, 4, false)*)mobi_attn_kernel_ptr_))(
        inputs[0].ptr<dtype_t>(), inputs[1].ptr<dtype_t>(), inputs[2].ptr<dtype_t>(),
        outputs[0].ptr<dtype_t>(), cargo_.B, cargo_.q_head, inputs[0].shape()[1],
        inputs[1].shape()[1], cargo_.D, cargo_.causal_mask);

  } else if (cargo_.q_head == 12 && cargo_.kv_head == 2 && cargo_.threads == 4 && !cargo_.hp_exp) {
    using dtype_t = DEF_FA2_NAME(12, 2, 4, false)::dtype_t;
    using acc_dtype_t = DEF_FA2_NAME(12, 2, 4, false)::acc_dtype_t;

    (*((DEF_FA2_NAME(12, 2, 4, false)*)mobi_attn_kernel_ptr_))(
        inputs[0].ptr<dtype_t>(), inputs[1].ptr<dtype_t>(), inputs[2].ptr<dtype_t>(),
        outputs[0].ptr<dtype_t>(), cargo_.B, cargo_.q_head, inputs[0].shape()[1],
        inputs[1].shape()[1], cargo_.D, cargo_.causal_mask);

  } else {
    MLLM_ERROR_EXIT(kError,
                    "cargo_.q_head({}), cargo_.kv_head({}), cargo_.threads({}), cargo_.hp_exp({}) "
                    "is not predefined in FA2 Op",
                    cargo_.q_head, cargo_.kv_head, cargo_.threads, cargo_.hp_exp);
  }
}

}  // namespace mllm::arm
