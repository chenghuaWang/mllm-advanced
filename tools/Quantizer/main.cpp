/**
 * @file main.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#include "mllm/Backends/Arm/Passes/KaiQuantizationPass.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include <memory>
#include <vector>
#include "fmt/base.h"
#include "fmt/ranges.h"
#include "mllm/Engine/Context.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Engine/CfgFile.hpp"
#include "mllm/Engine/ParameterReader.hpp"
#include "mllm/Utils/Argparse.hpp"
#include "mllm/IR/Passes/PassManager.hpp"
#include "tools/Quantizer/FlatModuleBuilder.hpp"

using namespace mllm;  // NOLINT

static void printLoaderMetaData(std::shared_ptr<ParameterLoader>& p) {
  std::vector<std::shared_ptr<TensorViewImpl>> _p{p->params().size(), nullptr};
  for (auto& item : p->params()) { _p[item.second->uuid()] = item.second; }
  for (auto& param : _p) {
    fmt::println("id: {}, name: {}, type: {}, shape: {}", param->uuid(), param->name(),
                 dataTypes2Str(param->dtype()), fmt::join(param->shape(), "x"));
  }
}

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

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& input_files =
      Argparse::add<std::string>("-i|--input").help("Input file path").meta("FILE").positional();
  auto& show_params = Argparse::add<bool>("-s|--show").help("Show parameters meta data");
  auto& cfg_file_path = Argparse::add<std::string>("-c|--config").help("QuantCfg file path");
  auto& write_to_file_path = Argparse::add<std::string>("-o|--output").help("Output file path");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!input_files.isSet()) {
    MLLM_ERROR_EXIT(kError, "No input file path provided");
    Argparse::printHelp();
    return -1;
  }

  auto loader = load(input_files.get());

  if (show_params.isSet()) {
    printLoaderMetaData(loader);
    return 0;
  }

  if (!cfg_file_path.isSet() || !write_to_file_path.isSet()) {
    MLLM_ERROR_EXIT(kError, "No cfg file path / output file path provided");
    Argparse::printHelp();
    return -1;
  }

  {
    MllmModelCfg cfg(cfg_file_path.get());
    std::vector<std::shared_ptr<BaseOp>> mllm_quantized_ops;
    auto ir_ctx = tools::createFlatModule(mllm_quantized_ops, loader, cfg);

    MLLM_INFO("Creating FAKE flat module for quantization");

    auto dump_printer = IRPrinter();
    ir_ctx->topLevelOp()->dump(dump_printer);

    for (auto& op_need_to_be_quantized : mllm_quantized_ops) {
      op_need_to_be_quantized->load(loader);
      MLLM_INFO("Planning {} op for quantization", op_need_to_be_quantized->name());
    }

    ir::PassManager pm(ir_ctx);
#if defined(__aarch64__)
#define MLLM_ON_ARM
    pm.reg(arm::createKaiQuantizationPass(cfg));
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
// Create other passes
#endif
    pm.run();

    // Update paramloader
    for (auto& op_quantized : mllm_quantized_ops) {
      auto this_op_params = op_quantized->params();
      for (auto& kv : this_op_params) { loader->params()[kv.first] = kv.second.impl(); }
    }

    mllm::write(write_to_file_path.get(), loader->params(), cfg.modelName());
    loader->params().clear();
  }
  ctx.shutdown();
}
