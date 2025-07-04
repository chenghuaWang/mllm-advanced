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
  std::string q_proj_name = "q_proj";
  std::string k_proj_name = "k_proj";
  std::string v_proj_name = "v_proj";
  std::string o_proj_name = "o_proj";
  int hidden_size = 1536;
  int intermediate_size = 8960;
  int heads_num = 12;
  int layers = 8;
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

class FakePayloadAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;

  int heads_num_;
  int head_size_;

 public:
  FakePayloadAttention() = default;

  explicit FakePayloadAttention(const std::string& name, const FakePayloadConfig& cfg) {
    selfAssignName(name);

    // init cfg
    heads_num_ = cfg.heads_num;
    head_size_ = cfg.hidden_size / heads_num_;

    // MHA
    q_proj_ = reg<nn::Linear>(cfg.q_proj_name, cfg.hidden_size, cfg.hidden_size, false);
    k_proj_ = reg<nn::Linear>(cfg.k_proj_name, cfg.hidden_size, cfg.hidden_size, false);
    v_proj_ = reg<nn::Linear>(cfg.v_proj_name, cfg.hidden_size, cfg.hidden_size, false);
    o_proj_ = reg<nn::Linear>(cfg.o_proj_name, cfg.hidden_size, cfg.hidden_size, false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // x shape is [S, D]
    auto x = inputs[0];

    auto S = x.shape()[0];

    // Q shape is [S, out_channel]
    auto q = q_proj_(x);
    q = q.view({S, heads_num_, head_size_});

    // K shape is [S, out_channel]
    auto k = k_proj_(x);
    k = k.view({S, heads_num_, head_size_});

    // V shape is [S, out_channel]
    auto v = v_proj_(x);
    v = v.view({S, heads_num_, head_size_});

    // LIMITATIONS:
    //
    // We just need approximate computation for simulate the actual workload.
    //
    // 1. We do not use RoPE for this module.
    // 2. KVCache is not output.
    // 3. Mask is not adopt
    // 4. softmax is not adopt
    // 5. rescale is not adopt
    auto w = nn::F::matmul(q, k, false, true);
    auto a = nn::F::matmul(w, v);
    a = a.view({S, heads_num_ * head_size_});
    auto o = o_proj_(a);
    return {o};
  }
};

class FakePayloadDecodeLayer final : public nn::Module {
  FakePayloadMLP mlp_;
  FakePayloadAttention self_attn_;

 public:
  FakePayloadDecodeLayer() = default;

  explicit FakePayloadDecodeLayer(const std::string& name, const FakePayloadConfig& cfg) {
    selfAssignName(name);
    mlp_ = reg<FakePayloadMLP>("mlp", cfg);
    self_attn_ = reg<FakePayloadAttention>("self_attn", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // LIMITATIONS:
    //
    // We just need approximate computation for simulate the actual workload.
    //
    // 1. RMSNorm is not used
    auto x = self_attn_(inputs[0])[0];
    auto tmp = x + inputs[0];
    x = mlp_(tmp)[0];
    x = x + tmp;
    return {x};
  }
};

class FakePayloadMultiMLP final : public nn::Module {
  nn::ModuleList<FakePayloadMLP> blocks_;

 public:
  FakePayloadMultiMLP() = default;

  FakePayloadMultiMLP(const std::string& name, const FakePayloadConfig& cfg) {
    selfAssignName(name);
    blocks_ = reg<nn::ModuleList<FakePayloadMLP>>("mlp", cfg.layers, cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto x = inputs[0];
    for (auto& block : blocks_.getList()) { x = block(x)[0]; }
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
    auto net = FakePayloadMultiMLP("model", cfg);
    net.to(kQNN);

    // clang-format off
    // 2. Make empty data.
    auto param_loader = std::make_shared<ParameterLoader>();
    for (int i = 0; i < cfg.layers; ++i) {
      auto id = std::to_string(i);
      auto gate_name = "model.mlp." + id + ".gate_proj.weight";
      auto up_name = "model.mlp." + id + ".up_proj.weight";
      auto down_name = "model.mlp." + id + ".down_proj.weight";
      param_loader->params()[gate_name] = Tensor::zeros({8960, 1536}, kFp16, kCPU).setMemType(kParams).setName(gate_name).impl();
      param_loader->params()[up_name] = Tensor::zeros({8960, 1536}, kFp16, kCPU).setMemType(kParams).setName(up_name).impl();
      param_loader->params()[down_name] = Tensor::zeros({1536, 8960}, kFp16, kCPU).setMemType(kParams).setName(down_name).impl();
    }
    // param_loader->params()["model.self_attn.q_proj.weight"] = Tensor::zeros({1536, 1536}, kFp16, kCPU).setMemType(kParams).setName("model.self_attn.q_proj.weight").impl();
    // param_loader->params()["model.self_attn.k_proj.weight"] = Tensor::zeros({1536, 1536}, kFp16, kCPU).setMemType(kParams).setName("model.self_attn.k_proj.weight").impl();
    // param_loader->params()["model.self_attn.v_proj.weight"] = Tensor::zeros({1536, 1536}, kFp16, kCPU).setMemType(kParams).setName("model.self_attn.v_proj.weight").impl();
    // param_loader->params()["model.self_attn.o_proj.weight"] = Tensor::zeros({1536, 1536}, kFp16, kCPU).setMemType(kParams).setName("model.self_attn.o_proj.weight").impl();
    net.load(param_loader);
    // clang-format on

    // 3. Trace IR
    auto ir_ctx = mllm::ir::trace(net, Tensor::empty({512, 1536}, kFp16, kQNN));

    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);

    // 4. Build Passes for compile, and Compile!
    ir::PassManager pm(ir_ctx);
    pm.reg(qnn::createQnnLoweringPipeline(qnn::QnnLoweringPipelineCfg{
        .tensor_readable_rename = true,
        .graphs_need_to_be_compiled = {"model"},
    }));
    pm.run();

    // 5. Check the compile IR
    ir_ctx->topLevelOp()->dump(dump_printer);

    // 6. Wrap Qnn object(The usage shown below is the low-level API in mllm)
    auto qnn_bk = std::static_pointer_cast<qnn::QnnBackend>(ctx.getBackend(kQNN));
    auto net_qnn_graph = qnn_bk->getCompiledQnnGraph("model");
    auto net_obj = qnn::QnnCompiledObj(
        net_qnn_graph, std::static_pointer_cast<qnn::QnnAllocator>(qnn_bk->getAllocator()));

    // 7. Run Qnn Object
    net_obj.allocRuntime();
    int64_t total_time = 0;
    auto test_start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1024 / 512; ++i) { net_obj.forward(); }
    // net_obj.forward();
    auto test_end_time = std::chrono::high_resolution_clock::now();
    total_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(test_end_time - test_start_time)
            .count();
    net_obj.freeRuntime();

    MLLM_INFO("Total Time {} ns", total_time);

    // 8. Set Params to normall for auto free memory.
    for (auto& p : param_loader->params()) { p.second->storage()->mem_type_ = kNormal; }
    param_loader->params().clear();
  }

  ctx.shutdown();
}