/**
 * @file tokenizer_qwen2.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Models/ds_qwen2/tokenization_ds_qwen2.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Preprocessor/Tokenizers/Unicode.hpp"
#include "mllm/Utils/Common.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace mllm::models {

bool deepSeekQwen2TokenizerMatchPattern(const std::wstring& str, size_t& pos,
                                        std::wstring& matched) {
  if (pos >= str.size()) return false;

  // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    if (pos + contraction.size() <= str.size()
        && str.compare(pos, contraction.size(), contraction) == 0) {
      matched = contraction;
      pos += contraction.size();
      return true;
    }
  }

  // 2. Match [^\r\n\p{L}\p{N}]?\p{L}+ (non-letter/digit followed by letters)
  {
    size_t original_pos = pos;
    bool has_prefix = false;
    matched.clear();

    // Check optional non-letter/digit prefix (excluding \r\n)
    if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r'
        && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
      has_prefix = true;
    }

    // Require at least one letter
    if (pos < str.size() && preprocessor::isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && preprocessor::isLetter(str[pos]));
      return true;
    } else {
      // Rollback if no letters after prefix
      if (has_prefix) {
        pos = original_pos;
        matched.clear();
      }
    }
  }

  // 3. Match \p{N} (digits)
  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  // 4. Match ?[^\s\p{L}\p{N}]+[\r\n]* (punctuation/symbols with optional space prefix)
  {
    size_t original_pos = pos;
    matched.clear();
    size_t start = pos;

    // Optional space
    if (str[pos] == L' ') { ++pos; }

    // Require at least one non-letter/digit/whitespace
    if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
        && !preprocessor::isDigit(str[pos])) {
      do {
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
               && !preprocessor::isDigit(str[pos]));

      // Capture from start (after optional space) to current pos
      matched = str.substr(start, pos - start);

      // Capture trailing newlines
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    } else {
      // Rollback if no symbols found
      pos = original_pos;
    }
  }

  // 5. Match \s*[\r\n]+ (newlines with leading whitespace)
  {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) ++pos;
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
  }

  // 6. Match \s+(?!\S) (whitespace not followed by non-space)
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    // Check if at end or followed by whitespace
    if (pos >= str.size() || std::iswspace(str[pos])) {
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
  }

  // 7. Match remaining whitespace
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

bool deepSeekQwen2Regex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (deepSeekQwen2TokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
  return true;
}

DeepSeekQwen2Tokenizer::DeepSeekQwen2Tokenizer(const std::string& file_path) {
  preprocessor::initLocal();
  preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
  for (auto& kv : bytes_2_unicode_dict_) {
    bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first});
  }
  bpe_.initFromSentencePieceJson(file_path);
  special_tokens_trie_.add(L"<｜end▁of▁sentence｜>");
  special_tokens_trie_.add(L"<｜begin▁of▁sentence｜>");
  special_tokens_trie_.add(L"<|quad_start|>");
  special_tokens_trie_.add(L"<|quad_end|>");
  special_tokens_trie_.add(L"<|vision_start|>");
  special_tokens_trie_.add(L"<|vision_end|>");
  special_tokens_trie_.add(L"<|vision_pad|>");
  special_tokens_trie_.add(L"<|image_pad|>");
  special_tokens_trie_.add(L"<|video_pad|>");
  special_tokens_trie_.add(L"<｜User｜>");
  special_tokens_trie_.add(L"<｜Assistant｜>");
}

std::vector<std::wstring> DeepSeekQwen2Tokenizer::_tokenize(const std::string& str) {
  std::vector<std::wstring> ret;
  std::vector<std::wstring> splitted;
  mllm::models::deepSeekQwen2Regex(str, splitted);
  for (const auto& s : splitted) {
    auto utf_8_str = preprocessor::wideString2Utf8String(s);
    std::wstring mapped_str;
    for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

    auto bpe_ts = bpe_._bpe(mapped_str);

    for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
  }

  return ret;
}

std::vector<std::wstring> DeepSeekQwen2Tokenizer::tokenize(const std::string& str) {
  auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
  std::vector<std::wstring> all_tokens;
  for (const auto& token : tokens) {
    if (special_tokens_trie_.isSpecialToken(token)) {
      all_tokens.emplace_back(token);
      continue;
    }
    auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
    all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
  }
  return all_tokens;
}

std::wstring DeepSeekQwen2Tokenizer::_detokenize(int64_t pos_idx) {
  return bpe_._lookup_inverse_vocab(pos_idx);
}

std::wstring DeepSeekQwen2Tokenizer::detokenize(int64_t pos_idx) {
  auto str = _detokenize(pos_idx);
  std::string utf_8_str;
  for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
  return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
}

Tensor DeepSeekQwen2Tokenizer::convert2Ids(const std::vector<std::wstring>& strs) {
  std::vector<int64_t> ids;
  ids.reserve(strs.size() + 1);
  ids.emplace_back(bpe_._lookup_vocab(L"<｜begin▁of▁sentence｜>"));
  for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
  Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                   .setMemType(kExtraInput)
                   .setName("qwen2-tokenizer-i0")
                   .alloc();

  auto ptr = ret.ptr<int64_t>();
  for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

  return ret;
}

}  // namespace mllm::models
