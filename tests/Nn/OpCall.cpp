#include "mllm/Backends/X86/X86Backend.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/Context.hpp"
#include <gtest/gtest.h>

using namespace mllm;

TEST(MllmNN, AddOpCall) {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::X86::createX86Backend());
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  auto a = Tensor::empty({128, 128}).alloc();
  auto b = Tensor::empty({128, 128}).alloc();
  auto c = a + b;

  a.print<float>();
  b.print<float>();
  c.print<float>();

  ctx.mem()->report();
}
