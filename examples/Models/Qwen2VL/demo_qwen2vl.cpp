/**
 * @file demo_qwen2vl.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Models/qwen2vl/modeling_qwen2vl.hpp"
#include "mllm/Models/qwen2vl/configuration_qwen2vl.hpp"
#include "mllm/Models/qwen2vl/tokenization_qwen2vl.hpp"

#include "mllm/Utils/Argparse.hpp"

using namespace mllm;  // NOLINT

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message.");
  auto& img_file_path = Argparse::add<std::string>("-i|--image")
                            .help("Input image ile path")
                            .meta("VISUAL_IMAGE_FILE");

  auto& model_file_path =
      Argparse::add<std::string>("-m|--model").help("Model files path").meta("MODEL_PATH");

  auto& tokenizer_file_path = Argparse::add<std::string>("-t|--tokenizer")
                                  .help("tokenizer files path")
                                  .meta("TOKENIZER_PATH");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!img_file_path.isSet()) {
    MLLM_ERROR_EXIT(kError, "No input visual image file provided");
    Argparse::printHelp();
    return -1;
  }

  if (!model_file_path.isSet()) {
    MLLM_ERROR_EXIT(kError, "No model file provided");
    Argparse::printHelp();
    return -1;
  }

  if (!tokenizer_file_path.isSet()) {
    MLLM_ERROR_EXIT(kError, "No tokenizer file provided");
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

  auto params = mllm::load(model_file_path.get());

  {
    auto tokenizer = models::Qwen2VLTokenizer(tokenizer_file_path.get(), 56 * 56, 28 * 28 * 256);
    auto inputs = tokenizer.convertMessage({
        .prompt = "Describe this image.",
        .img_file_path = img_file_path.get(),
    });
    auto qwen2vl_cfg = models::Qwen2VLConfig();
    auto vit = models::Qwen2VisionTransformerPretrainedModel("visual", qwen2vl_cfg);
    vit.load(params);

    inputs.grid_thw.print<int>();

    auto img_embedding = vit(inputs.image, inputs.grid_thw)[0];

    img_embedding.print<float>();
  }

  ctx.shutdown();
}
