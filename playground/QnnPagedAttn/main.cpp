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

class QnnPagedAttnKernelMMA0 final : public nn::Module {
 public:
  ~QnnPagedAttnKernelMMA0() {
    // Let mllm to release this tensor by mark this tensor is a normal tensor instead of a param
    // tensor.
    causal_mask_.setMemType(kNormal);
  }

  QnnPagedAttnKernelMMA0() = default;

  explicit QnnPagedAttnKernelMMA0(const std::string& name, int32_t q_head, int32_t kv_head,
                                  int32_t sys_token_length, int32_t visual_token_block_length) {
    selfAssignName(name);
  }

  void verify(const std::vector<Tensor>& inputs) {
    MLLM_RT_ASSERT_EQ(inputs.size(), 6);
    auto& query = inputs[0];
    auto& key_sys = inputs[1];
    auto& key_v0 = inputs[2];
    auto& key_v1 = inputs[3];
    auto& key_v2 = inputs[4];
    auto& key_v3 = inputs[5];

    // Not support batch
    MLLM_RT_ASSERT_EQ(key_sys.shape().size(), 3);
    MLLM_RT_ASSERT_EQ(key_v0.shape().size(), 3);
    MLLM_RT_ASSERT_EQ(key_v1.shape().size(), 3);
    MLLM_RT_ASSERT_EQ(key_v2.shape().size(), 3);
    MLLM_RT_ASSERT_EQ(key_v3.shape().size(), 3);
  }

  void initCausalMask() {
    causal_mask_ =
        Tensor::zeros({visual_token_block_length_, visual_token_block_length_}, kFp16, kCPU)
            .setMemType(kParams)
            .setName("mma0.causal_mask");

    for (int r = 0; r < visual_token_block_length_; ++r) {
      for (int c = 0; c < visual_token_block_length_; ++c) {
        if (r < c) continue;
        (*causal_mask_.offsettedPtr<float16_t>({r, c})) = FloatInfo<kFp16>::min;
      }
    }

    causal_mask_.print<float16_t>();
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // Verify first before gen IR
    verify(inputs);

    // The first input is the query
    auto& query = inputs[0];
    auto key_sys = inputs[1];
    auto key_v0 = inputs[2];
    auto key_v1 = inputs[3];
    auto key_v2 = inputs[4];
    auto key_v3 = inputs[5];

    // For GQA. Repeat with q_head / kv_head times
    auto repeat_times = q_head_ / kv_head_;
    if (repeat_times > 1) {
      key_sys = key_sys.repeat(repeat_times, 0);
      key_v0 = key_v0.repeat(repeat_times, 0);
      key_v1 = key_v1.repeat(repeat_times, 0);
      key_v2 = key_v2.repeat(repeat_times, 0);
      key_v3 = key_v3.repeat(repeat_times, 0);
    }

    return {nn::F::matmul(query, key_sys, false, true),
            nn::F::matmul(query, key_v0, false, true),
            nn::F::matmul(query, key_v1, false, true),
            nn::F::matmul(query, key_v2, false, true),
            nn::F::matmul(query, key_v3, false, true),
            nn::F::matmul(query, query, false, true) + causal_mask_};
  }

 private:
  Tensor causal_mask_;
  int32_t q_head_ = 12;
  int32_t kv_head_ = 6;
  int32_t sys_token_length_ = 15;
  int32_t visual_token_block_length_ = 64;
};

class QnnPagedAttnKernelSoftmax final : public nn::Module {
 public:
  QnnPagedAttnKernelSoftmax() = default;

  explicit QnnPagedAttnKernelSoftmax(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // TODO cast to fp32 to perform softmax and then cast to fp16
    return {};
  }
};

class QnnPagedAttnKernelMMA1 final : public nn::Module {
 public:
  QnnPagedAttnKernelMMA1() = default;

  explicit QnnPagedAttnKernelMMA1(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // TODO
    return {};
  }
};

class QnnPagedAttnKernelScale final : public nn::Module {
 public:
  QnnPagedAttnKernelScale() = default;

  explicit QnnPagedAttnKernelScale(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // TODO
    return {};
  }
};

class QnnPagedAttnKernelRescale final : public nn::Module {
 public:
  QnnPagedAttnKernelRescale() = default;

  explicit QnnPagedAttnKernelRescale(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // TODO
    return {};
  }
};

class QnnPagedAttnKernel final : public nn::Module {
  QnnPagedAttnKernelMMA0 mma0_;
  QnnPagedAttnKernelSoftmax softmax_;
  QnnPagedAttnKernelMMA1 mma1_;
  QnnPagedAttnKernelScale scale_;
  QnnPagedAttnKernelRescale rescale_;

 public:
  QnnPagedAttnKernel() = default;

  explicit QnnPagedAttnKernel(const std::string& name) { selfAssignName(name); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // TODO
    return {};
  }
};

void testMMA0(MllmEngineCtx& ctx) {
  auto paged_attn_kernel_mma0 = QnnPagedAttnKernelMMA0("mma0", 12, 6, 15, 64);
  paged_attn_kernel_mma0.to(kQNN);
  paged_attn_kernel_mma0.initCausalMask();
  auto ir_ctx =
      mllm::ir::trace(paged_attn_kernel_mma0, Tensor::empty({12, 64, 128}, kFp16, kQNN),  // query
                      Tensor::empty({6, 15, 128}, kFp16, kQNN),                           // key_sys
                      Tensor::empty({6, 64, 128}, kFp16, kQNN),                           // key_v0
                      Tensor::empty({6, 64, 128}, kFp16, kQNN),                           // key_v1
                      Tensor::empty({6, 64, 128}, kFp16, kQNN),                           // key_v2
                      Tensor::empty({6, 64, 128}, kFp16, kQNN)                            // key_v3
      );
  auto dump_printer = IRPrinter();
  ir_ctx->topLevelOp()->dump(dump_printer);

  ir::PassManager pm(ir_ctx);
  pm.reg(qnn::createQnnLoweringPipeline(qnn::QnnLoweringPipelineCfg{
      .tensor_readable_rename = true,
      .graphs_need_to_be_compiled = {"mma0"},
  }));
  pm.run();

  ir_ctx->topLevelOp()->dump(dump_printer);
}

int main() {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::arm::createArmBackend());
  ctx.registerBackend(mllm::qnn::createQnnBackend());
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  { testMMA0(ctx); }

  ctx.shutdown();
}
