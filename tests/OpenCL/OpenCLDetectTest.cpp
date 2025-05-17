/**
 * @file OpenCLDetectTest.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-17
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#include "mllm/Backends/OpenCL/OpenCLBackend.hpp"

using namespace mllm;  // NOLINT

int main() {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::opencl::createOpenCLBackend());
}