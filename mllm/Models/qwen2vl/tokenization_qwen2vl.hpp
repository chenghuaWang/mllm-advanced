/**
 * @file tokenization_qwen2vl.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Models/qwen2vl/configuration_qwen2vl.hpp"
#include "mllm/Preprocessor/Tokenizers/AutoTokenizer.hpp"
#include "mllm/Preprocessor/Tokenizers/BPE.hpp"

#include "mllm/Models/qwen2vl/image_preprocessor_qwen2vl.hpp"

#include <vector>
#include <unordered_map>

namespace mllm::models {

// we need to handle this:
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
bool qwen2VLTokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched);

bool qwen2VLRegex(const std::string& str, std::vector<std::wstring>& splitted);

struct Qwen2VLMessage {
  std::string prompt;
  std::string img_file_path;
  static inline std::string message_template =
      "<|im_start|>system\nYou are a helpful "
      "assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|vision_pad|><|vision_end|>{{{"
      "prompt}}}<|im_end|>\n<|im_start|>assistant\n";
};

class Qwen2VLTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit Qwen2VLTokenizer(const std::string& file_path, int32_t min_patches = 56 * 56,
                            int32_t max_patches = 28 * 28 * 256);

  std::vector<std::wstring> _tokenize(const std::string& str) override;

  std::vector<std::wstring> tokenize(const std::string& str) override;

  std::wstring _detokenize(int64_t pos_idx) override;

  std::wstring detokenize(int64_t pos_idx) override;

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override;

  Qwen2VLForCausalLMOutputPast convertMessage(const Qwen2VLMessage& message);

 private:
  // For image only.
  Qwen2VLImagePreprocessor image_preprocessor_;

  // For text
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models
