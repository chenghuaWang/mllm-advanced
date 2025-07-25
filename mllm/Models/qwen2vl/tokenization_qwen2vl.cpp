/**
 * @file tokenization_qwen2vl.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Models/qwen2vl/tokenization_qwen2vl.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Models/qwen2vl/configuration_qwen2vl.hpp"
#include "mllm/Preprocessor/Tokenizers/Unicode.hpp"
#include "mllm/Utils/Common.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mllm::models {

// we need to handle this:
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
bool qwen2VLTokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
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

bool qwen2VLRegex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (qwen2VLTokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
  return true;
}

Qwen2VLTokenizer::Qwen2VLTokenizer(const std::string& file_path, int32_t min_patches,
                                   int32_t max_patches)
    : image_preprocessor_(min_patches, max_patches) {
  preprocessor::initLocal();
  preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
  for (auto& kv : bytes_2_unicode_dict_) {
    bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first});
  }
  bpe_.initFromSentencePieceJson(file_path);
  special_tokens_trie_.add(L"<|endoftext|>");
  special_tokens_trie_.add(L"<|im_start|>");
  special_tokens_trie_.add(L"<|im_end|>");
  special_tokens_trie_.add(L"<|object_ref_start|>");
  special_tokens_trie_.add(L"<|object_ref_end|>");
  special_tokens_trie_.add(L"<|box_start|>");
  special_tokens_trie_.add(L"<|box_end|>");
  special_tokens_trie_.add(L"<|quad_start|>");
  special_tokens_trie_.add(L"<|quad_end|>");
  special_tokens_trie_.add(L"<|vision_start|>");
  special_tokens_trie_.add(L"<|vision_end|>");
  special_tokens_trie_.add(L"<|vision_pad|>");
  special_tokens_trie_.add(L"<|image_pad|>");
  special_tokens_trie_.add(L"<|video_pad|>");
}

std::vector<std::wstring> Qwen2VLTokenizer::_tokenize(const std::string& str) {
  std::vector<std::wstring> ret;
  std::vector<std::wstring> splitted;
  mllm::models::qwen2VLRegex(str, splitted);
  for (const auto& s : splitted) {
    auto utf_8_str = preprocessor::wideString2Utf8String(s);
    std::wstring mapped_str;
    for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

    auto bpe_ts = bpe_._bpe(mapped_str);

    for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
  }

  return ret;
}

std::vector<std::wstring> Qwen2VLTokenizer::tokenize(const std::string& str) {
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

std::wstring Qwen2VLTokenizer::_detokenize(int64_t pos_idx) {
  return bpe_._lookup_inverse_vocab(pos_idx);
}

std::wstring Qwen2VLTokenizer::detokenize(int64_t pos_idx) {
  auto str = _detokenize(pos_idx);
  std::string utf_8_str;
  for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
  return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
}

Tensor Qwen2VLTokenizer::convert2Ids(const std::vector<std::wstring>& strs) {
  std::vector<int64_t> ids;
  ids.reserve(strs.size());
  for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
  Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                   .setMemType(kExtraInput)
                   .setName("qwen2-tokenizer-i0")
                   .alloc();

  auto ptr = ret.ptr<int64_t>();
  for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

  return ret;
}

Qwen2VLForCausalLMOutputPast Qwen2VLTokenizer::convertMessage(const Qwen2VLMessage& message) {
  // process prompt
  auto applied_string = Qwen2VLMessage::message_template;
  size_t pos = applied_string.find("{{{prompt}}}");
  applied_string.replace(pos, 12, message.prompt);

  MLLM_INFO("{}", applied_string);

  // process image
  auto [img, grid_thw] = image_preprocessor_(message.img_file_path);

  // process sequence
  auto sequence_str = tokenize(applied_string);
  std::vector<int64_t> ids;
  ids.reserve(sequence_str.size());
  for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

  // Get img's token nums
  auto grid_t = grid_thw.ptr<int32_t>()[0];
  auto grid_h = grid_thw.ptr<int32_t>()[1];
  auto grid_w = grid_thw.ptr<int32_t>()[2];
  int32_t img_token_nums = grid_t * grid_h * grid_w;
  img_token_nums /= 4;

  // Find img_pad_token_ids pos and insert img_token_nums-1 times after this token
  auto img_pad_token_ids = bpe_._lookup_vocab(L"<|vision_pad|>");
  {
    auto it = std::find(ids.begin(), ids.end(), img_pad_token_ids);  // NOLINT
    ids.insert(it + 1, img_token_nums - 1, img_pad_token_ids);
  }

  // Get sequence Tensor
  Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                        .setMemType(kNormal)
                        .setName("qwen2-tokenizer-i0")
                        .alloc();

  auto ptr = sequence.ptr<int64_t>();
  for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

  return Qwen2VLForCausalLMOutputPast{
      .sequence = sequence, .img = img, .grid_thw = grid_thw, .has_visual = true};
}

}  // namespace mllm::models
