#include "mllm/Backends/CUDA/CUDABackend.hpp"
#include "mllm/Backends/X86/X86Backend.hpp"
#include "mllm/Engine/Context.hpp"

#include <gtest/gtest.h>

using namespace mllm;

TEST(MllmCUDA, Context) {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::cuda::createCUDABackend());
  ctx.registerBackend(mllm::X86::createX86Backend());
  ctx.mem()->initBuddyCtx(kCUDA);
  ctx.mem()->initOC(kCUDA);
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    auto a = Tensor::empty({1024, 1024}, kBF16, kCUDA).alloc();
    auto b = Tensor::empty({1024, 1024}, kBF16, kCUDA).alloc();
    auto c = a + b;
  }

  ctx.shutdown();
}