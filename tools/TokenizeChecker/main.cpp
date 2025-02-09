/**
 * @file main.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <iostream>
#include <string>
#include "mllm/Models/qwen2/tokenization_qwen2.hpp"

int main() {
  mllm::models::Qwen2Tokenizer tokenizer(
      "/media/wch/D/mllm-all/mllm-models/DeepSeek-R1-Distill-Qwen-1.5B/tokenizer.json");
  std::string res;
  std::getline(std::cin, res);
  tokenizer._tokenize(res);
}
