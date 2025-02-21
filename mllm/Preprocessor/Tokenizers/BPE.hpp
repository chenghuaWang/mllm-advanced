/**
 * @file BPE.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <functional>

using json = nlohmann::json;

namespace mllm::preprocessor {

struct BPEPairHash {
  std::size_t operator()(const std::pair<std::wstring, std::wstring>& key) const {
    std::size_t h1 = std::hash<std::wstring>{}(key.first + key.second);
    return h1;
  }
};

class BPE {
 public:
  // BPE can accept sentence piece's json foramt.
  bool initFromSentencePieceJson(const std::string& file_path);

  std::vector<std::wstring> _bpe(const std::wstring& token);

  long _lookup_vocab(const std::wstring& token);

  std::wstring _lookup_inverse_vocab(long idx);

 private:
  std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> _get_pairs(
      const std::vector<std::wstring>& word);

  std::unordered_map<std::wstring, long> vocab_;
  std::unordered_map<long, std::wstring> vocab_inverse_;
  std::unordered_map<std::pair<std::wstring, std::wstring>, long, BPEPairHash> bpe_ranks_;
};

}  // namespace mllm::preprocessor
