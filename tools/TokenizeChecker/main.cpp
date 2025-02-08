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
#include "mllm/Models/qwen2/tokenizer_qwen2.hpp"
#include "mllm/Preprocessor/Tokenizers/Unicode.hpp"

int main() {
  mllm::preprocessor::initLocal();
  std::string res;
  std::getline(std::cin, res);
  std::vector<std::wstring> splited;
  mllm::models::qwen2Regex(res, splited);
  for (auto& s : splited) { std::wcout << s << std::endl; }
}
