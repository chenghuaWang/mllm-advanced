#include <mllm/Engine/Context.hpp>
#include <mllm/Backends/Arm/ArmBackend.hpp>
#include <mllm/Backends/QNN/QnnBackend.hpp>

#include <mllm/Nn/F/F.hpp>
#include <mllm/Nn/Module.hpp>

#include <mllm/IR/IR.hpp>
#include <mllm/Utils/IRPrinter.hpp>
#include <mllm/IR/Passes/PassManager.hpp>
#include <mllm/Backends/QNN/Runtime/QnnCompiledObj.hpp>
#include <mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp>

using namespace mllm;  // NOLINT

class FakePayloadNet final : public nn::Module {
 public:
  FakePayloadNet() = default;

  explicit FakePayloadNet(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    return {nn::F::matmul(inputs[0], inputs[1])};
  }
};

int main() {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::arm::createArmBackend());
  ctx.registerBackend(mllm::qnn::createQnnBackend());
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    // 1. Create Model
    auto net = FakePayloadNet("fake_model");
    net.to(kQNN);
    auto ir_ctx = mllm::ir::trace(net, Tensor::empty({1, 1, 128, 128}, kFp32, kQNN),
                                  Tensor::empty({1, 1, 128, 128}, kFp32, kQNN));

    ir::PassManager pm(ir_ctx);

    // 2. Build Passes for compile, and Compile!
    pm.reg(qnn::createQnnLoweringPipeline(qnn::QnnLoweringPipelineCfg{
        .tensor_readable_rename = true,
        .graphs_need_to_be_compiled = {"fake_model"},
    }));
    pm.run();

    // 3. Check the compile IR
    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);

    // 4. Wrap Qnn object(The usage shown below is the low-level API in mllm)
    auto qnn_bk = std::static_pointer_cast<qnn::QnnBackend>(ctx.getBackend(kQNN));
    auto net_qnn_graph = qnn_bk->getCompiledQnnGraph("fake_model");
    auto net_obj = qnn::QnnCompiledObj(
        net_qnn_graph, std::static_pointer_cast<qnn::QnnAllocator>(qnn_bk->getAllocator()));

    // 5. Run Qnn Object
    net_obj.allocRuntime();
    net_obj.forward();
    net_obj.freeRuntime();
  }

  ctx.shutdown();
}