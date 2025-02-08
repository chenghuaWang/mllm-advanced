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

std::wstring utf8string2WideString(const std::string& str) {
  std::wstring w_ret_string;
  for (unsigned int i = 0; i < str.size();) {
    auto byte = static_cast<unsigned char>(str[i]);
    if ((byte & 0x80) == 0) {
      // 1-byte character
      w_ret_string += static_cast<wchar_t>(byte);
      ++i;
    } else if ((byte & 0xE0) == 0xC0) {
      // 2-byte character
      if (i + 1 < str.size()) {
        wchar_t wc =
            (static_cast<wchar_t>(byte & 0x1F) << 6) | (static_cast<wchar_t>(str[i + 1] & 0x3F));
        w_ret_string += wc;
        i += 2;
      } else {
        break;
      }
    } else if ((byte & 0xF0) == 0xE0) {
      // 3-byte character
      if (i + 2 < str.size()) {
        wchar_t wc = (static_cast<wchar_t>(byte & 0x0F) << 12)
                     | (static_cast<wchar_t>(str[i + 1] & 0x3F) << 6)
                     | (static_cast<wchar_t>(str[i + 2] & 0x3F));
        w_ret_string += wc;
        i += 3;
      } else {
        break;
      }
    } else if ((byte & 0xF8) == 0xF0) {
      // 4-byte character
      if (i + 3 < str.size()) {
        wchar_t wc = (static_cast<wchar_t>(byte & 0x07) << 18)
                     | (static_cast<wchar_t>(str[i + 1] & 0x3F) << 12)
                     | (static_cast<wchar_t>(str[i + 2] & 0x3F) << 6)
                     | (static_cast<wchar_t>(str[i + 3] & 0x3F));
        w_ret_string += wc;
        i += 4;
      } else {
        break;
      }
    } else {
      // Invalid UTF-8 sequence
      ++i;
    }
  }
  return w_ret_string;
}

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
