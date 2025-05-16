/**
 * @file main.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstdint>
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Log.hpp"

#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Utils/Argparse.hpp"
#include "mllm/Models/ds_qwen2/tokenization_ds_qwen2.hpp"

using namespace mllm;  // NOLINT

void initMllmCtx() {
  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);
}

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message.");
  auto& se_json_fp = Argparse::add<std::string>("-j|--json").help("SentencePiece json file path.");
  auto& merge_fp = Argparse::add<std::string>("-m|--merge").help("Merge file path.");
  auto& model_type = Argparse::add<std::string>("-t|--type").help("Model Type.");
  auto& input_str = Argparse::add<std::string>("-i|--input_str").help("Input string for testing.");

  MLLM_WARN("The '\\n'(All Escape Characters) inputs to mllm-tokenize-checker will not be treated "
            "as a new line but two characters('\\' and 'n'). This is a known issue due to the "
            "default behaviour of the terminal. Pls use `mllm-tokenize-checker -t <model name> -j "
            "<json path> -i $'ANSI-C reference style \\n'` instead. This issue is only exists in "
            "the terminal. tokenizer.tokenize(...) works fine with Escape Characters.");

  Argparse::parse(argc, argv);

  if (help.isSet() || !model_type.isSet() || !input_str.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  initMllmCtx();

  auto model_name = model_type.get();

  if (model_name == "ds-qwen2") {
    MLLM_INFO("{}", input_str.get());
    auto tokenizer = std::make_shared<models::DeepSeekQwen2Tokenizer>(se_json_fp.get());
    auto t = tokenizer->convert2Ids(tokenizer->tokenize(input_str.get()));
    t.print<int64_t>();
  }
}
