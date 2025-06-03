/**
 * @file QnnDetectTest.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#include "mllm/Backends/QNN/QnnBackend.hpp"

using namespace mllm;  // NOLINT

int main() {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::qnn::createQnnBackend());
}