#include <mllm/Engine/Context.hpp>
#include <mllm/Backends/Arm/ArmBackend.hpp>
#include <mllm/Backends/QNN/QnnBackend.hpp>

#include <mllm/Engine/ParameterReader.hpp>

#include <mllm/Nn/F/F.hpp>
#include <mllm/Nn/Module.hpp>

#include <mllm/Nn/Layers/SiLU.hpp>
#include <mllm/Nn/Layers/Linear.hpp>

#include <mllm/IR/IR.hpp>
#include <mllm/Utils/IRPrinter.hpp>
#include <mllm/IR/Passes/PassManager.hpp>
#include <mllm/Backends/QNN/Runtime/QnnCompiledObj.hpp>
#include <mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp>

#include <chrono>

using namespace mllm;  // NOLINT

struct FakePayloadConfig {
  std::string gate_proj_name = "gate_proj";
  std::string up_proj_name = "up_proj";
  std::string down_proj_name = "down_proj";
  int hidden_size = 1536;
  int intermediate_size = 8960;
};

class FakePayloadMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  FakePayloadMLP() = default;

  explicit FakePayloadMLP(const std::string& name, const FakePayloadConfig& cfg) {
    selfAssignName(name);
    gate_proj_ = reg<nn::Linear>(cfg.gate_proj_name, cfg.hidden_size, cfg.intermediate_size, false);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>(cfg.up_proj_name, cfg.hidden_size, cfg.intermediate_size, false);
    down_proj_ = reg<nn::Linear>(cfg.down_proj_name, cfg.intermediate_size, cfg.hidden_size, false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
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
    auto cfg = FakePayloadConfig();
    auto net = FakePayloadMLP("mlp", cfg);
    net.to(kQNN);

    // clang-format off
    // 2. Make empty data.
    auto param_loader = std::make_shared<ParameterLoader>();
    param_loader->params()["mlp.gate_proj.weight"] = Tensor::zeros({8960, 1536}, kFp16, kCPU).setMemType(kParams).setName("mlp.gate_proj.weight").impl();
    param_loader->params()["mlp.up_proj.weight"] = Tensor::zeros({8960, 1536}, kFp16, kCPU).setMemType(kParams).setName("mlp.up_proj.weight").impl();
    param_loader->params()["mlp.down_proj.weight"] = Tensor::zeros({1536, 8960}, kFp16, kCPU).setMemType(kParams).setName("mlp.down_proj.weight").impl();
    net.load(param_loader);
    // clang-format on

    // 3. Trace IR
    auto ir_ctx = mllm::ir::trace(net, Tensor::empty({1, 1536}, kFp16, kQNN));
    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);

    // 4. Build Passes for compile, and Compile!
    ir::PassManager pm(ir_ctx);
    pm.reg(qnn::createQnnLoweringPipeline(qnn::QnnLoweringPipelineCfg{
        .tensor_readable_rename = true,
        .graphs_need_to_be_compiled = {"mlp"},
    }));
    pm.run();

    // 5. Check the compile IR
    ir_ctx->topLevelOp()->dump(dump_printer);

    // 6. Wrap Qnn object(The usage shown below is the low-level API in mllm)
    auto qnn_bk = std::static_pointer_cast<qnn::QnnBackend>(ctx.getBackend(kQNN));
    auto net_qnn_graph = qnn_bk->getCompiledQnnGraph("mlp");
    auto net_obj = qnn::QnnCompiledObj(
        net_qnn_graph, std::static_pointer_cast<qnn::QnnAllocator>(qnn_bk->getAllocator()));

    // 7. Run Qnn Object
    net_obj.allocRuntime();
    net_obj.forward();
    net_obj.freeRuntime();
  }

  ctx.shutdown();
}