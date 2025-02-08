/**
 * @file tokenizer_qwen2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Preprocessor/Tokenizers/Unicode.hpp"
#include <vector>

namespace mllm::models {

// we need to handle this:
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
bool qwen2TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched);

bool qwen2Regex(const std::string& str, std::vector<std::wstring>& splited);

}  // namespace mllm::models
