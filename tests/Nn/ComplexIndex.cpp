#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Nn/F/F.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/Context.hpp"
#include <gtest/gtest.h>

using namespace mllm; // NOLINT

TEST(MllmNN, AddOpCall) {
  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    auto x = Tensor::ones({1, 4, 8}, kFp32, kCPU);
    auto [x0, x1, x2, x3] = nn::F::split<4>(x, /*split_size_or_sections=*/1, /*dim=*/1);
    x.print<float>();
    // [1, 1, 8]
    x0.print<float>();
    x1.print<float>();
    x2.print<float>();
    x3.print<float>();
    // view
    x0 = x0.view({1, 1, -1, 4});
    x0.print<float>();
    int cnt = 0;
    TiledTensor(x0).complexLoops<float>(
        {
            "_0",
            "_1",
            "_2",
            "_3",
        },
        [&](float* ptr, const std::vector<int32_t>& offsets) -> void { *ptr = cnt++; });
    x0.print<float>();
    x.print<float>();
    TiledTensor(x0).parallelLoops<float>(
        3, [](float* ptr, int b_stride, const std::vector<int32_t>& left_dims) -> void {
          *ptr = 666.F;
        });
    x0.print<float>();
    x.print<float>();
  }

  ctx.mem()->report();
}
