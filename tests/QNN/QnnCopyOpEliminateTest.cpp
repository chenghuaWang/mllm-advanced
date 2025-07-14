#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#include "mllm/Backends/QNN/QnnBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Engine/Context.hpp"

#include "mllm/Nn/Module.hpp"
#include "mllm/Nn/F/F.hpp"
#include "mllm/Nn/Planning.hpp"

#include "mllm/IR/IR.hpp"
#include "mllm/Utils/IRPrinter.hpp"
#include "mllm/IR/Passes/PassManager.hpp"
#include "mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp"

#include "mllm/IR/Passes/CopyOpEliminatePass.hpp"

#include "mllm/Backends/QNN/Runtime/QnnCompiledObj.hpp"

using namespace mllm;  // NOLINT

class ExampleNet final : public nn::Module {
 public:
  ExampleNet() = default;

  explicit ExampleNet(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto x0 = inputs[0];
    auto& x1 = inputs[1];
    auto dst = inputs[2];

    auto should_be_eliminated = x0 + x1;
    nn::planning::copy(dst, should_be_eliminated);
    return {dst};
  }
};

int main() {
  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
  ctx.registerBackend(mllm::qnn::createQnnBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    auto net = ExampleNet("ExampleNet");
    net.to(kQNN);
    auto ir_ctx = mllm::ir::trace(net, Tensor::empty({1024, 1024}, kFp32, kQNN),
                                  Tensor::empty({1024, 1024}, kFp32, kQNN),
                                  Tensor::empty({1024, 1024}, kFp32, kQNN));

    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);

    ir::PassManager pm(ir_ctx);
    pm.reg(ir::createCopyOpEliminatePass());
    pm.run();

    ir_ctx->topLevelOp()->dump(dump_printer);
  }

  ctx.shutdown();
}