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

#include <memory>
#include "mllm/IR/Passes/Pass.hpp"
#include "mllm/Backends/QNN/Passes/QnnTensorNamingPass.hpp"

namespace mllm::qnn {

std::vector<std::shared_ptr<ir::Pass>> createQnnLoweringPipeline(/*TODO cfg struct*/);

}  // namespace mllm::qnn