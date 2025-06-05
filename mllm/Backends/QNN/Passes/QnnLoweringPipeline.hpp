/**
 * @file QnnLoweringPipeline.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <vector>
#include <string>
#include <memory>
#include "mllm/IR/Passes/Pass.hpp"
#include "mllm/Backends/QNN/Passes/QnnTensorNamingPass.hpp"

namespace mllm::qnn {

struct QnnLoweringPipelineCfg {
  bool tensor_readable_rename = true;
  std::vector<std::string> graphs_need_to_be_compiled;
};

std::vector<std::shared_ptr<ir::Pass>> createQnnLoweringPipeline(
    const QnnLoweringPipelineCfg& cfg = QnnLoweringPipelineCfg{.tensor_readable_rename = true});

}  // namespace mllm::qnn