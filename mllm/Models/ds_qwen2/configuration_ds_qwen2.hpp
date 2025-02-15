/**
 * @file configuration_ds_qwen2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>

struct QWenConfig {
  std::string gate_proj_name;
  std::string up_proj_name;
  std::string down_proj_name;

  int hidden_size = 1536;
  int intermediate_size = 8960;
};
