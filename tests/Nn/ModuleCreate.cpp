#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Nn/Layers/Linear.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Engine/Context.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class InnerModule : public nn::Module {
 public:
  std::vector<Tensor> forward(std::vector<Tensor>& inputs) override {}
};

class ExampleModule : public nn::Module {
  nn::Linear linear_0;
  nn::Linear linear_1;
  InnerModule inner_module;

 public:
  ExampleModule() {
    selfAssginName("module");
    inner_module = reg<InnerModule>("inner_module");
    linear_0 = reg<nn::Linear>(
        "linear_0", LinearOpCargo{.in_channels = 1024, .out_channels = 2048, .bias = true}
                        .setInputsDtype(LinearOpCargo::InputsPos::kWeight, kFp16)
                        .setInputsDtype(LinearOpCargo::InputsPos::kInput, kFp16)
                        .setOutputsDtype(LinearOpCargo::OutputsPos::kOutput, kFp16));
    linear_1 = reg<nn::Linear>("linear_1",
                               LinearOpCargo{.in_channels = 2048, .out_channels = 128, .bias = true}
                                   .setInputsDtype(LinearOpCargo::InputsPos::kWeight, kFp16)
                                   .setInputsDtype(LinearOpCargo::InputsPos::kInput, kFp16)
                                   .setOutputsDtype(LinearOpCargo::OutputsPos::kOutput, kFp16));
  }

  std::vector<Tensor> forward(std::vector<Tensor>& inputs) override {
    auto x = inputs[0];
    x = linear_0(x);
    x = linear_1(x);
    return {x};
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

  auto e = ExampleModule();
  e.print();
}