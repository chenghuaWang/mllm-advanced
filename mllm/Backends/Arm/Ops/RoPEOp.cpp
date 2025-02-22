/**
 * @file RoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/RoPEOp.hpp"
#include "mllm/Core/AOps/RoPEOp.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/Arm/Kernels/rope.hpp"

namespace mllm::arm {

ArmRoPEOp::ArmRoPEOp(const RoPEOpCargo& cargo) : RoPEOp(cargo) {}

void ArmRoPEOp::load(std::shared_ptr<ParameterLoader>& ploader) {
  auto& ctx = MllmEngineCtx::instance();
  if (ctx.mem()->hasGlobalTensor("__global_rope_sin")
      && ctx.mem()->hasGlobalTensor("__global_rope_cos")) {
    sin_ = ctx.mem()->getGlobalTensor("__global_rope_sin");
    cos_ = ctx.mem()->getGlobalTensor("__global_rope_cos");
    return;
  }

  // init sin and cos
  switch (cargo_.type) {
    case RoPETypes::kLlama2: {
      sin_ =
          Tensor::empty({(size_t)cargo_.max_position_embeddings, (size_t)cargo_.dims}, kFp32, kCPU)
              .setMemType(kGlobal)
              .setName("__global_rope_sin")
              .alloc();
      ctx.mem()->regGlobalTensor(sin_);
      cos_ =
          Tensor::empty({(size_t)cargo_.max_position_embeddings, (size_t)cargo_.dims}, kFp32, kCPU)
              .setMemType(kGlobal)
              .setName("__global_rope_cos")
              .alloc();
      ctx.mem()->regGlobalTensor(cos_);
      precompute_normal_hf_sin_cos(cargo_.max_position_embeddings, cargo_.dims, cargo_.theta,
                                   sin_.ptr<float>(), cos_.ptr<float>());
      break;
    }
    default: NYI("ArmRoPEOp find a unsupported rope type."); break;
  }
}

void ArmRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Only support [B, H, S, D] layout
  auto X = inputs[0];
  auto Y = outputs[0];

  MLLM_RT_ASSERT_EQ(X.shape().size(), 4);

  switch (cargo_.type) {
    case RoPETypes::kLlama2: {
      auto shape = X.shape();
      auto B = shape[0];
      auto H = shape[1];
      auto S = shape[2];
      auto D = shape[3];
      for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
          normal_hf_rope(X.offsettedPtr<float>({b, h, 0, 0}), Y.offsettedPtr<float>({b, h, 0, 0}),
                         sin_.ptr<float>(), cos_.ptr<float>(), cur_seq_cnt_, S, D);
        }
      }
      break;
    }
    default: NYI("ArmRoPEOp find a unsupported rope type."); break;
  }

  // inputs is [B, H, S, D]
  cur_seq_cnt_ += inputs[0].shape()[2];
}

}  // namespace mllm::arm
