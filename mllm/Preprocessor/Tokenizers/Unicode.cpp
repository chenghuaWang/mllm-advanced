/**
 * @file Unicode.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Preprocessor/Tokenizers/Unicode.hpp"
#include <vector>
#include <algorithm>

namespace mllm::preprocessor {

void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t>& dict) {
  std::vector<std::wint_t> bs((ord(L"~") - ord(L"!") + 1) + (ord(L"¬") - ord(L"¡") + 1)
                              + (ord(L"ÿ") - ord(L"®") + 1));

  int cnt = 0;
  for (std::wint_t i = ord(L"!"); i <= ord(L"~"); ++i) { bs[cnt++] = i; }
  for (std::wint_t i = ord(L"¡"); i <= ord(L"¬"); ++i) { bs[cnt++] = i; }
  for (std::wint_t i = ord(L"®"); i <= ord(L"ÿ"); ++i) { bs[cnt++] = i; }

  std::vector<std::wint_t> cs(bs.size());
  for (int i = 0; i < bs.size(); ++i) { cs[i] = bs[i]; }

  int n = 0;
  for (std::wint_t b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.emplace_back(b);
      cs.emplace_back(256 + n);
      ++n;
    }
  }

  std::vector<wchar_t> cs_chars(cs.size());
  for (int i = 0; i < cs.size(); ++i) { cs_chars[i] = chr(cs[i]); }
  for (int i = 0; i < bs.size(); ++i) { dict.insert({bs[i], cs_chars[i]}); }
}

}  // namespace mllm::preprocessor
