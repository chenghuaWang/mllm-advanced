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
#include "mllm/Preprocessor/Tokenizers/AutoTokenizer.hpp"
#include "mllm/Preprocessor/Tokenizers/BPE.hpp"
#include <vector>
#include <unordered_map>

namespace mllm::models {

// we need to handle this:
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
bool deepSeekQwen2TokenizerMatchPattern(const std::wstring& str, size_t& pos,
                                        std::wstring& matched);

bool deepSeekQwen2Regex(const std::string& str, std::vector<std::wstring>& splited);

class DeepSeekQwen2Tokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit DeepSeekQwen2Tokenizer(const std::string& file_path);

  std::vector<std::wstring> _tokenize(const std::string& str) override;

  std::vector<std::wstring> tokenize(const std::string& str) override;

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override;

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
};

}  // namespace mllm::models
