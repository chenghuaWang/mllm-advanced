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
#include "mllm/Models/AutoLLM.hpp"
#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Models/qwen2vl/modeling_qwen2vl.hpp"
#include "mllm/Models/qwen2vl/image_preprocessor_qwen2vl.hpp"

#include "mllm/Utils/Argparse.hpp"

using namespace mllm;  // NOLINT

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message.");
  auto& img_file_path = Argparse::add<std::string>("-i|--image")
                            .help("Input image ile path")
                            .meta("VISUAL_IMAGE_FILE");
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

  auto& ctx = MllmEngineCtx::instance();
#if defined(MLLM_ON_X86)
  ctx.registerBackend(mllm::X86::createX86Backend());
#endif
#if defined(MLLM_ON_ARM)
  ctx.registerBackend(mllm::arm::createArmBackend());
#endif
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    models::Qwen2VLImagePreprocessor img_preprocessor;
    auto [image_tensor, grid_thw] = img_preprocessor(img_file_path.get());
    image_tensor.print<float>();
  }

  ctx.shutdown();
}