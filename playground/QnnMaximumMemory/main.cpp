#include <arm_neon.h>

#include <mllm/Utils/Common.hpp>
#include <mllm/Engine/Context.hpp>
#include <mllm/Backends/Arm/ArmBackend.hpp>
#include <mllm/Backends/QNN/QnnBackend.hpp>

#include <mllm/Nn/F/F.hpp>
#include <mllm/Nn/Module.hpp>

#include <mllm/IR/IR.hpp>
#include <mllm/Utils/IRPrinter.hpp>
#include <mllm/IR/Passes/PassManager.hpp>
#include <mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp>

using namespace mllm;  // NOLINT

class HugeMem final : public nn::Module {
 public:
  HugeMem() = default;

  explicit HugeMem(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto x1 = inputs[0];
    auto x2 = inputs[1];
    auto o = x1 + x2;
    return {o};
  }
};

int main() {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::arm::createArmBackend());
  ctx.registerBackend(mllm::qnn::createQnnBackend());
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    auto net = HugeMem("huge");
    net.to(kQNN);
    auto ir_ctx = mllm::ir::trace(
        net, Tensor::empty({1024, 1024}, kFp32, kQNN), Tensor::empty({1024, 1024}, kFp32, kQNN)

    );
    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);

    ir::PassManager pm(ir_ctx);
    pm.reg(qnn::createQnnLoweringPipeline(qnn::QnnLoweringPipelineCfg{
        .tensor_readable_rename = true,
        .graphs_need_to_be_compiled = {"huge"},
    }));
    pm.run();

    ir_ctx->topLevelOp()->dump(dump_printer);
  }

  ctx.shutdown();
}