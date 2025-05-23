/**
 * @file AutoTokenizer.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// Ths json lib is head only. Include all functionall <nlohmann/json.hpp> will increase the compile
// time. Hence, <nlohmann/json_fwd.hpp> is provided for decrease compile time.
//
// json_fwd.hpp:
// Used for forward declaring the nlohmann::json type, suitable for scenarios where only the type
// needs to be declared, reducing compilation time and dependencies.
//
// json.hpp:
// Contains the full implementation of the JSON library, suitable for scenarios where JSON data
// needs to be manipulated.
#include <nlohmann/json_fwd.hpp>
using json = nlohmann::json;

#include "mllm/Core/Tensor.hpp"

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>

namespace mllm::preprocessor {

// split text to tokens.
// > Trie.addSpecial("<|im_start|>")
// > Trie.split("<|im_start|>Hello world!")
//
// will give: ["<|im_start|>","Hello world!"]
class Trie {
  struct TrieNode {
    std::unordered_map<wchar_t, std::unique_ptr<TrieNode>> children;
    bool is_end = false;
  };

 public:
  void add(const std::wstring& word);

  void update(const std::vector<std::wstring>& words);

  // I use FSA to implement the split function.
  std::vector<std::wstring> split(const std::wstring& text);

  bool isSpecialToken(const std::wstring& token);

 private:
  std::unique_ptr<TrieNode> root_ = std::make_unique<TrieNode>();
  std::unordered_set<std::wstring> special_tokens_;
};

class AutoTokenizer {
 public:
  void addSpecialToken(const std::wstring& special_token);

  virtual std::vector<std::wstring> _tokenize(const std::string& str) = 0;

  virtual std::vector<std::wstring> tokenize(const std::string& str) = 0;

  virtual std::wstring _detokenize(long pos_idx) = 0;

  virtual std::wstring detokenize(long pos_idx) = 0;

  virtual Tensor convert2Ids(const std::vector<std::wstring>& strs) = 0;

 protected:
  Trie special_tokens_trie_;
};

}  // namespace mllm::preprocessor
