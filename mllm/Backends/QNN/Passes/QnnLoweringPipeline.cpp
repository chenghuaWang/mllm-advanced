/**
 * @file QnnLoweringPipeline.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Passes/QnnGraphBuildPass.hpp"
#include "mllm/Backends/QNN/Passes/QnnGraphInlinePass.hpp"
#include "mllm/Backends/QNN/Passes/QnnMarkIOTensorPass.hpp"
#include "mllm/Backends/QNN/Passes/QnnTensorNamingPass.hpp"
#include "mllm/Backends/QNN/Passes/QnnAnonymousOpNamingPass.hpp"
#include "mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp"

namespace mllm::qnn {

std::vector<std::shared_ptr<ir::Pass>> createQnnLoweringPipeline(
    const QnnLoweringPipelineCfg& cfg) {
  std::vector<std::shared_ptr<ir::Pass>> ret;

  if (cfg.tensor_readable_rename) { ret.emplace_back(createQnnTensorNamingPass()); }

  // NOTE: Notice the Pass execution order !!!
  // NOTE: Notice the Pass execution order !!!
  // NOTE: Notice the Pass execution order !!!

  ret.emplace_back(createQnnGraphInlinePass(cfg.graphs_need_to_be_compiled));
  ret.emplace_back(createQnnAnonymousOpNamingPass());

  // Do final compile
  ret.emplace_back(createQnnMarkIOTensorPass());  // After all graph modification
  ret.emplace_back(createQnnGraphBuildPass(cfg.graphs_need_to_be_compiled));

  return ret;
}

}  // namespace mllm::qnn