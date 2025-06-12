/**
 * @file CfgFile.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <fstream>
#include "mllm/Engine/CfgFile.hpp"
#include "mllm/Utils/Common.hpp"
#include <nlohmann/json.hpp>

namespace mllm {

MllmModelCfg::MllmModelCfg(const std::string& file_path) {
  using namespace nlohmann;  // NOLINT

  std::ifstream file(file_path);

  if (!file.is_open()) { MLLM_ERROR_EXIT(kError, "Failed to open config file: {}", file_path); }
  json_ = json::parse(file);
}

std::string MllmModelCfg::opImplType(const std::string& op_name) const {
  if (!json_.contains("Ops") || !json_["Ops"].contains(op_name)
      || !json_["Ops"][op_name].contains("implType")) {
    MLLM_ERROR_EXIT(kError, "Missing required field in config for operator: {}", op_name);
  }

  return json_["Ops"][op_name]["implType"].get<std::string>();
}

}  // namespace mllm
