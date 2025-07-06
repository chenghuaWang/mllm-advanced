#if defined(__aarch64__)
#define MLLM_ON_ARM
#include "mllm/Backends/Arm/ArmBackend.hpp"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MLLM_ON_X86
#include "mllm/Backends/X86/X86Backend.hpp"
#endif

#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Argparse.hpp"

#include "mllm/Preprocessor/Visual/Image.hpp"

using namespace mllm;  // NOLINT
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

  auto& image_files_path = Argparse::add<std::string>("-i|--image")
                               .help("Input image file path")
                               .meta("FILE")
                               .positional();
  Argparse::parse(argc, argv);

  {
    auto image = Image::open(image_files_path.get());
    auto image_tensor = image.tensor();
    image_tensor.print<float>();
  }

  ctx.shutdown();
}