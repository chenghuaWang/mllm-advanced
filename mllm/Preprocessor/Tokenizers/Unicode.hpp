/**
 * @file Unicode.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cwchar>
#include <unordered_map>

namespace mllm::preprocessor {

inline std::wint_t ord(const wchar_t* str) { return static_cast<std::wint_t>(*str); }

inline wchar_t chr(std::wint_t value) { return static_cast<wchar_t>(value); }

// same with gpt2.bytes_to_unicode
//
// same with qwen2.bytes_to_unicode
//
/*
Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to
whitespace/control characters the bpe code barfs on.

The reversible bpe codes work on unicode strings. This means you need a large # of unicode
characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token
dataset you end up needing around 5K for decent coverage. This is a significant percentage of
your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
unicode strings.
*/
void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t>& dict);

}  // namespace mllm::preprocessor
