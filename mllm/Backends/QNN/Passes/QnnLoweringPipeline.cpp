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
#include "mllm/Backends/QNN/Passes/QnnLoweringPipeline.hpp"

namespace mllm::qnn {

std::vector<std::shared_ptr<ir::Pass>> createQnnLoweringPipeline(/*TODO cfg struct*/) {
  return {createQnnTensorNamingPass()};
}

}  // namespace mllm::qnn