/**
 * @file CfgFile.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// Process Config Files of
// 1. Mllm Engine
// 2. Model
// ...

#include <nlohmann/json_fwd.hpp>

namespace mllm {

struct MllmEngineCfg {};

struct MllmModelCfg {
  // Op's Weights and it's inputs and outputs dtype
};

}  // namespace mllm
