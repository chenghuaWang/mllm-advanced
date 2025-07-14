#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Nn/Module.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Nn/Layers/KVCache.hpp"

#include <gtest/gtest.h>

using namespace mllm;  // NOLINT

class ExampleModule : public nn::Module {
 public:
  nn::KVCache x_cache_;

  ExampleModule() {
    selfAssignName("module");
    x_cache_ = reg<nn::KVCache>("x_cache", 1, 8, 2, kFp32, 1024, 2);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    return {x_cache_(inputs[0])};
  }
};

TEST(MllmNN, NestedModuleCreate) {
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
    auto e = ExampleModule();
    e.print();

    // [B, H, S, D]
    auto x = Tensor::ones({1, 1, 1, 8});
    x.print<float>();
    auto y = e(x)[0];
    y.print<float>();
    auto two = x + x;
    y = e(two)[0];
    y.print<float>();
    auto eight = two * two * two;
    y = e(eight)[0];
    y.print<float>();
    y = y.contiguous();
    y.print<float>();
  }

  ctx.shutdown();
}