#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/Context.hpp"

#include "mllm/Nn/Planning.hpp"

#include <gtest/gtest.h>

using namespace mllm;  // NOLINT

TEST(MllmNN, WriteToTest) {
  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);
#if defined(MLLM_ON_ARM)
  {
    auto c = Tensor::empty({128, 128}, kFp16).alloc();  // Prealloc
    auto a = Tensor::ones({128, 128}, kFp16);
    auto b = Tensor::ones({128, 128}, kFp16);

    // Write to
    nn::planning::write2(c).from(a + b);

    a.print<float16_t>();
    b.print<float16_t>();
    c.print<float16_t>();
  }
#endif
  {
    auto c = Tensor::empty({128, 128}, kFp32).alloc();  // Prealloc
    auto a = Tensor::ones({128, 128}, kFp32);
    auto b = Tensor::ones({128, 128}, kFp32);

    // Write to
    nn::planning::write2(c).from(a + b);

    a.print<float>();
    b.print<float>();
    c.print<float>();
  }
  ctx.mem()->report();
}
