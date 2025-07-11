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

std::string MllmModelCfg::modelName() const {
  if (!json_.contains("ModelName")) {
    MLLM_ERROR_EXIT(kError, "Missing required field: ModelName in config");
  }
  return json_["ModelName"].get<std::string>();
}

std::string MllmModelCfg::opImplType(const std::string& op_name) const {
  if (!json_.contains("Ops") || !json_["Ops"].contains(op_name)
      || !json_["Ops"][op_name].contains("implType")) {
    MLLM_ERROR_EXIT(kError, "Missing required field in config for operator: {}", op_name);
  }

  return json_["Ops"][op_name]["implType"].get<std::string>();
}

std::string MllmModelCfg::opType(const std::string& op_name) const {
  if (!json_.contains("Ops") || !json_["Ops"].contains(op_name)
      || !json_["Ops"][op_name].contains("type")) {
    MLLM_ERROR_EXIT(kError, "Missing required field in config for operator: {}", op_name);
  }

  return json_["Ops"][op_name]["type"].get<std::string>();
}

std::vector<std::string> MllmModelCfg::opNames() const {
  std::vector<std::string> result;
  if (json_.contains("Ops")) {
    for (const auto& [key, value] : json_["Ops"].items()) { result.push_back(key); }
  }
  return result;
}

std::vector<std::string> MllmModelCfg::paramNames() const {
  std::vector<std::string> result;
  if (json_.contains("Params")) {
    for (const auto& [key, value] : json_["Params"].items()) { result.push_back(key); }
  }
  return result;
}

DataTypes MllmModelCfg::paramDtype(const std::string& param_name) const {
  if (json_["Params"].contains(param_name) && json_["Params"][param_name].contains("dtype")) {
    if (json_["Params"][param_name]["dtype"] == "Fp32") return DataTypes::kFp32;
    if (json_["Params"][param_name]["dtype"] == "Fp16") return DataTypes::kFp16;
  }
  return DataTypes::kFp32;
}

nlohmann::json& MllmModelCfg::rawJson() { return json_; }

}  // namespace mllm
