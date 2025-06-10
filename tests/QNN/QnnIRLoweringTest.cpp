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

#include "mllm/IR/IR.hpp"
#include "mllm/Utils/IRPrinter.hpp"
#include "mllm/IR/Passes/PassManager.hpp"
#include "mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp"

using namespace mllm;  // NOLINT

class OneOpNet final : public nn::Module {
 public:
  OneOpNet() = default;

  explicit OneOpNet(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    return {nn::F::matmul(inputs[0], inputs[1])};
  }
};

int main(int argc, char* argv[]) {
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
    auto net = OneOpNet("OneOpNet");
    net.to(kQNN);
    auto ir_ctx = mllm::ir::trace(net, Tensor::empty({1, 1, 128, 1024}, kFp32, kQNN),
                                  Tensor::empty({1, 1, 1024, 128}, kFp32, kQNN));

    // Lowering
    ir::PassManager pm(ir_ctx);
    pm.reg(qnn::createQnnLoweringPipeline(qnn::QnnLoweringPipelineCfg{
        .tensor_readable_rename = true,
        .graphs_need_to_be_compiled = {"OneOpNet"},
    }));
    pm.run();

    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);
  }

  ctx.shutdown();
}