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

#include <nlohmann/json.hpp>
#include <vector>
#include "mllm/Core/DataTypes.hpp"

namespace mllm {

class MllmEngineCfg {
  // TODO
};

class MllmModelCfg {
 public:
  explicit MllmModelCfg(const std::string& file_path);

  [[nodiscard]] std::string modelName() const;

  /**
   * @brief Parses the operator configuration section from the loaded JSON file.
   *
   * This function processes the "Ops" section in the configuration file, which defines
   * custom operator implementations and quantization settings for the model. The
   * configuration allows specifying different implementations for different
   * operators in the model graph.
   *
   * @details
   * The configuration follows this format:
   * @code
   * {
   *   "Ops": {
   *     "model.linear_0": {
   *       "implType": "KaiLinear_fp16_fp16_fp16p_mxk_kxn"
   *     }
   *   }
   * }
   * @endcode
   *
   * This enables fine-grained control over:
   * - Which operator implementation to use (e.g., different kernels for different architectures)
   * - Quantization parameters (through naming conventions in the impl string)
   * - Operator-specific optimizations
   *
   * The parsed configuration is typically used by the model to determine which
   * operator implementation to instantiate for each node in the computation graph.
   */
  [[nodiscard]] std::string opImplType(const std::string& op_name) const;

  [[nodiscard]] std::string opType(const std::string& op_name) const;

  [[nodiscard]] std::vector<std::string> opNames() const;

  [[nodiscard]] std::vector<std::string> paramNames() const;

  [[nodiscard]] DataTypes paramDtype(const std::string& param_name) const;

 protected:
  nlohmann::json json_;
};

}  // namespace mllm
