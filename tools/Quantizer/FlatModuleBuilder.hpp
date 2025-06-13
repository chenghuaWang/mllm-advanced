/**
 * @file FlatModuleBuilder.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief Helper Class to generate a flat module for IR to process params.
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include "mllm/IR/Node.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Engine/CfgFile.hpp"
#include "mllm/Engine/ParameterReader.hpp"

namespace mllm::tools {

std::shared_ptr<ir::IRContext> createFlatModule(
    std::vector<std::shared_ptr<BaseOp>>& mllm_quantized_ops,
    std::shared_ptr<ParameterLoader>& param_loader, MllmModelCfg& cfg);

}