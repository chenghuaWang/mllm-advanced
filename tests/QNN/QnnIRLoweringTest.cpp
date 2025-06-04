#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Engine/Context.hpp"
#include "mllm/IR/IR.hpp"
#include "mllm/Utils/IRPrinter.hpp"
#include "mllm/Models/ds_qwen2/modeling_ds_qwen2.hpp"
#include "mllm/Models/ds_qwen2/configuration_ds_qwen2.hpp"

#include "mllm/IR/Passes/PassManager.hpp"
#include "mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp"

#include "mllm/Utils/Argparse.hpp"

using namespace mllm;  // NOLINT
int main(int argc, char* argv[]) {
  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  auto& model_files = Argparse::add<std::string>("-m|--model")
                          .help("Input model ile path")
                          .meta("FILE")
                          .positional();

  Argparse::parse(argc, argv);

  {
    mllm::models::QWenConfig cfg;
    mllm::models::QWenForCausalLM llm(cfg);
    llm.load(mllm::load(model_files.get()));
    llm.to(kCPU);
    auto ir_ctx = mllm::ir::trace(llm, Tensor::empty({1, 128}, kInt64));

    // Lowering
    ir::PassManager pm(ir_ctx);
    pm.reg(qnn::createQnnLoweringPipeline());
    pm.run();

    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);
  }

  ctx.shutdown();
}