/**
 * @file demo_ds_qwen2.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <iostream>
#include "mllm/Engine/Context.hpp"
#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Models/ds_qwen2/modeling_ds_qwen2.hpp"
#include "mllm/Models/ds_qwen2/tokenization_ds_qwen2.hpp"
#include "mllm/Models/ds_qwen2/configuration_ds_qwen2.hpp"

#include "mllm/Utils/Argparse.hpp"

using namespace mllm;

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message.");
  auto& model_files = Argparse::add<std::string>("-m|--model")
                          .help("Input model ile path")
                          .meta("FILE")
                          .positional();
  auto& se_json_fp = Argparse::add<std::string>("-j|--json").help("SentencePiece json file path.");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!model_files.isSet()) {
    MLLM_ERROR_EXIT(kError, "No input model file provided");
    Argparse::printHelp();
    return -1;
  }

  if (!se_json_fp.isSet()) {
    MLLM_ERROR_EXIT(kError, "No input vocab file provided");
    Argparse::printHelp();
    return -1;
  }

  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);
  ctx.mem()->updateCacheSizeList(kCPU, {
                                           // TODO run benchmark to get the best cache size
                                       });

  // perf or not.
  ctx.perf_ = false;

  {
    mllm::models::DeepSeekQwen2Tokenizer tokenizer(se_json_fp.get());
    mllm::models::QWenConfig cfg;
    mllm::models::QWenForCausalLM model(cfg);
    model.print();

    auto loader = mllm::load(model_files.get());
    model.load(loader);

    auto input = tokenizer.convert2Ids(tokenizer.tokenize("hello, what's u name?"));

    auto o = model(input);

    std::wcout << tokenizer.detokenize(o[0]) << std::endl;
  }

  ctx.shutdown();
}
