/**
 * @file main.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <iostream>
#include <memory>

#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Log.hpp"
#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Server/Generator.hpp"
#include "mllm/Utils/Argparse.hpp"
#include "mllm/Server/RequestServer.hpp"
#include "mllm/Utils/Common.hpp"

// ds_qwen2
#include "mllm/Models/ds_qwen2/modeling_ds_qwen2.hpp"
#include "mllm/Models/ds_qwen2/tokenization_ds_qwen2.hpp"

using namespace mllm;  // NOLINT

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message.");
  auto& port = Argparse::add<int>("-p|--port").help("Port to listen on.").meta("PORT");
  auto& model_tag =
      Argparse::add<std::string>("-m|--model").help("Model tag to use.").meta("MODEL");
  auto& model_param_path = Argparse::add<std::string>("-mp|--model-param-path")
                               .help("Model param path.")
                               .meta("MODEL_PARAM_PATH");
  auto& model_vocab_path = Argparse::add<std::string>("-vp|--model-vocab-path")
                               .help("Model vocab path.")
                               .meta("MODEL_VOCAB_PATH");
  auto& model_config_path = Argparse::add<std::string>("-c|--model-config-path")
                                .help("Model config path.")
                                .meta("MODEL_CONFIG_PATH");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  MLLM_RT_ASSERT(model_tag.isSet());

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

  auto generator = std::make_shared<MllmStreamModelGenerator>(nullptr);

  if (model_tag.get() == "DeepSeek-R1-Distill-Qwen-1.5B-fp32") {
    mllm::models::DeepSeekQwen2Tokenizer tokenizer(model_vocab_path.get());
    mllm::models::QWenConfig cfg(model_config_path.get());
    mllm::models::QWenForCausalLM llm(cfg);
    llm.print();
    auto loader = mllm::load(model_param_path.get());
    llm.load(loader);

    // set eos id
    generator->eos_token_id_ = cfg.eos_token_id;

    // model
    generator->setModel(&llm);

    // how to build prompt
    generator->setBuildPromptCallback(
        [](const std::vector<std::pair<std::string, std::string>>& prompts) -> std::string {
          if (prompts.size() != 1) {
            MLLM_ASSERT_EXIT(kError, "Currently support only one prompt.");
          }

          return "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>" + prompts[0].second
                 + "<｜Assistant｜>";
        });

    // encode
    generator->setTokenizerEncodeCallback([&tokenizer](const std::string& str) -> Tensor {
      auto t = tokenizer.convert2Ids(tokenizer.tokenize(str));
      return t;
    });

    // decode
    generator->setTokenizerDecodeCallback(
        [&tokenizer](int64_t pos_idx) { return tokenizer.detokenize(pos_idx); });

    MllmRequestServer server(port.get());
    server.setMllmStreamModelGenerator(generator);
    server.setModelTag(model_tag.get());
    server.start();
    MLLM_INFO("Press q to quit.");
    char ch;
    while (std::cin >> ch) {
      if (ch == 'q') {
        server.stop();
        break;
      }
    }
  } else {
    NYI("The Model tag {} is not supported.", model_tag.get());
  }

  ctx.shutdown();
  ctx.mem()->report();
  MLLM_INFO("BYE");
}
